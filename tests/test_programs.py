"""tinyllm.tg.programs: a session's compiled programs, kept on disk between starts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).parent.parent


def test_program_directory_comes_from_the_environment_or_the_cache_directory() -> None:
    from tinyllm.tg import programs

    assert programs.directory({"TINYLLM_PROGRAM_CACHE": "0"}) is None
    assert programs.directory({"TINYLLM_PROGRAM_CACHE": "off"}) is None
    assert programs.directory({"TINYLLM_PROGRAM_CACHE": "/somewhere/else"}) == Path("/somewhere/else")
    assert programs.directory({"XDG_CACHE_HOME": "/cache"}) == Path("/cache/tinyllm/programs")
    assert programs.directory({}) == Path("~/.cache").expanduser() / "tinyllm" / "programs"


def test_model_files_lists_every_part_of_a_split_model(tmp_path: Path) -> None:
    from tinyllm.tg import programs

    parts = [tmp_path / f"m[x]-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    for p in [*parts, tmp_path / "m[x]-00001-of-00002.gguf", tmp_path / "other.gguf"]:
        p.write_bytes(b"gguf")
    assert programs.model_files(parts[0]) == parts
    assert programs.model_files(tmp_path / "other.gguf") == [tmp_path / "other.gguf"]


def test_session_key_follows_the_files_the_settings_and_the_environment(tmp_path: Path, monkeypatch: Any) -> None:
    from tinyllm.tg import programs

    model = tmp_path / "model.gguf"
    model.write_bytes(b"0123")
    monkeypatch.delenv("TINYLLM_KERNELS", raising=False)
    key, parts = programs.session_key([model], {"chunk": 32, "max_context": 512})
    assert programs.session_key([model], {"max_context": 512, "chunk": 32})[0] == key
    assert parts["files"][0][:2] == ["model.gguf", 4]
    assert programs.session_key([model], {"chunk": 64, "max_context": 512})[0] != key
    monkeypatch.setenv("TINYLLM_PROGRAM_CACHE", str(tmp_path))  # where they are kept does not change what they are
    assert programs.session_key([model], {"chunk": 32, "max_context": 512})[0] == key
    monkeypatch.setenv("TINYLLM_KERNELS", "none")
    assert programs.session_key([model], {"chunk": 32, "max_context": 512})[0] != key
    monkeypatch.delenv("TINYLLM_KERNELS")
    model.write_bytes(b"01234")
    assert programs.session_key([model], {"chunk": 32, "max_context": 512})[0] != key


class _Layer:
    state: Any  # set by the first call
    kernel: str
    hook: Any

    def __init__(self) -> None:
        from tinyllm.tg import Tensor

        self.weight = Tensor(np.arange(16, dtype=np.float32).reshape(4, 4) / 16).contiguous().realize()


class _Session:
    """What `UpstreamSession` does with its programs, on a model of one layer whose first call makes its state."""

    def __init__(self, directory: Path, scale: int = 1) -> None:
        from tinyllm.tg import programs

        self.programs = programs.Programs(self, directory)
        self.layer, self.scale = _Layer(), scale
        self.step = self.programs.jit("step", self._step)
        self.programs.identify("one layer", [], {"scale": scale})
        self.programs.restore()

    def _step(self, x: Any) -> Any:
        from tinyllm.tg import Tensor

        layer = self.layer
        if not hasattr(layer, "state"):  # as the model's blocks make their caches
            layer.state = Tensor.zeros(4).contiguous().realize()
            layer.kernel = "dense"
        out = (x @ layer.weight * self.scale + layer.state).contiguous()
        Tensor.realize(out, layer.state.assign(out))
        return out

    def run(self, calls: int = 4) -> list[np.ndarray]:
        from tinyllm.tg import Tensor

        outs = []
        for i in range(calls):
            x = Tensor(np.full((1, 4), i + 1, dtype=np.float32)).contiguous().realize()
            outs.append(self.step(x).numpy().copy())
        return outs


def test_a_second_session_loads_programs_attributes_and_zeroed_state(tmp_path: Path) -> None:
    built = _Session(tmp_path)
    assert not built.programs.restored and "none saved" in built.programs.note
    want = built.run()
    assert built.programs.pending() == [] and built.programs.save(), built.programs.note
    files = sorted(p.name for p in next(tmp_path.iterdir()).iterdir())
    assert files == ["manifest.json", "programs.pkl"]

    loaded = _Session(tmp_path)
    assert loaded.programs.restored and loaded.programs.loaded == ["step"], loaded.programs.note
    assert loaded.step.cnt == 2  # nothing to build
    # what the first call set on the layer is back: the choice it stored, and its state, zeroed as the call made it
    assert loaded.layer.kernel == "dense"
    assert np.array_equal(loaded.layer.state.numpy(), np.zeros(4, dtype=np.float32))
    assert loaded.layer.state.uop.base is not built.layer.state.uop.base
    got = loaded.run()
    assert all(np.array_equal(a, b) for a, b in zip(want, got, strict=True))
    assert not loaded.programs.save()  # loaded programs are not written again

    other = _Session(tmp_path, scale=2)  # other settings: another key, built again
    assert not other.programs.restored


def test_programs_are_not_loaded_from_a_file_others_can_write_and_not_saved_with_what_cannot_be(tmp_path: Path) -> None:
    built = _Session(tmp_path)
    built.run()
    assert built.programs.save()
    file = next(tmp_path.iterdir()) / "programs.pkl"
    os.chmod(file, 0o666)
    refused = _Session(tmp_path)
    assert not refused.programs.restored and "someone else" in refused.programs.note
    assert not hasattr(refused.layer, "state")  # the session is untouched and builds its programs

    again = _Session(tmp_path / "second")
    again.run()
    again.layer.hook = lambda x: x  # something a first call left that no file can hold
    assert not again.programs.save() and again.programs.note.startswith("not saved")
    assert not (tmp_path / "second").exists() or not any((tmp_path / "second").rglob("programs.pkl"))


def test_a_program_is_captured_by_its_first_call_once_another_of_its_function_is(tmp_path: Path) -> None:
    from tinyllm.tg import Tensor

    def x(value: float) -> Any:
        return Tensor(np.full((1, 4), value, dtype=np.float32)).contiguous().realize()

    want = _Session(tmp_path / "plain").run(3)  # one program, built in two calls
    s = _Session(tmp_path / "early")
    other = s.programs.jit("step again", s._step, s.step)  # the same function: nothing left for a first call to make
    got = [s.step(x(1)).numpy().copy(), s.step(x(2)).numpy().copy()]
    assert s.step.cnt == 2 and s.step.linears == 1 and not s.step.early
    got.append(other(x(3)).numpy().copy())
    assert other.early and not other.again and other.cnt == 2 and other.captured is not None
    assert all(np.array_equal(a, b) for a, b in zip(want, got, strict=True))
    assert np.array_equal(other(x(4)).numpy(), _Session(tmp_path / "plain").run(4)[3])  # and it replays

    early = s.programs.jit("before", s._step, s.programs.jit("never run", s._step))
    early(x(5))
    assert not early.early and early.cnt == 1  # the other program is not captured: an ordinary first call


def test_a_first_call_that_made_something_is_not_kept_as_the_capture(tmp_path: Path) -> None:
    from tinyllm.tg import Tensor, programs

    tables: dict[int, Any] = {}

    def scaled(x: Any) -> Any:
        n = int(x.shape[1])
        if n not in tables:  # made once per width: captured, it would be made again at every call
            tables[n] = Tensor(np.arange(1, n + 1, dtype=np.float32)).contiguous().realize()
        out = (x * tables[n]).contiguous()
        out.realize()
        return out

    def x(n: int, value: float) -> Any:
        return Tensor(np.full((1, n), value, dtype=np.float32)).contiguous().realize()

    p = programs.Programs(None)
    narrow = p.jit("narrow", scaled)
    wide = p.jit("wide", scaled, narrow)
    for value in (1.0, 2.0):
        assert np.array_equal(narrow(x(4, value)).numpy(), value * np.arange(1, 5, dtype=np.float32)[None])
    assert narrow.linears == 1
    # the wide program's first call makes its table: two realizes where the narrow program's capture had one
    assert np.array_equal(wide(x(8, 3.0)).numpy(), 3.0 * np.arange(1, 9, dtype=np.float32)[None])
    assert wide.again and not wide.early and wide.cnt == 1 and wide.captured is None
    assert np.array_equal(wide(x(8, 4.0)).numpy(), 4.0 * np.arange(1, 9, dtype=np.float32)[None])  # captures
    assert wide.cnt == 2 and wide.linears == 1
    assert np.array_equal(wide(x(8, 5.0)).numpy(), 5.0 * np.arange(1, 9, dtype=np.float32)[None])


def test_a_process_loads_the_programs_another_built_and_computes_the_same(dev_model: Path) -> None:
    """The small model on the CPU through `tools/program_cache_check.py`: a second process finds the first one's
    programs, builds none, and its logits after two prompts and while decoding are the first one's bit for bit."""
    cmd = [sys.executable, "tools/program_cache_check.py", str(dev_model), "--device", "cpu", "--ctx", "512"]
    cmd += ["--chunk", "16", "--prompt", "40", "--new", "20", "--steps", "3"]
    env = {k: v for k, v in os.environ.items() if k not in ("TINYLLM_PROGRAM_CACHE", "JITBEAM")}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=900, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "0 failure(s)" in proc.stdout and "FAIL" not in proc.stdout


@pytest.mark.gpu
def test_flash_next_program_captured_by_its_first_call_is_the_one_two_calls_build() -> None:
    """Qwen3.8-Flash-Next `UD-IQ4_XS` with its MTP block (`tools/program_build_check.py`): the 768-token prefill
    program captured by its first call, its tuned kernels lowered in worker processes, gives the logits and state
    of the same program built in two calls, bit for bit."""
    from tinyllm.server.serve import find_mtp
    from tinyllm.weights.resolve import find_model

    try:
        path = find_model("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if find_mtp(path) is None:
        pytest.skip(f"no MTP block next to {path.name}")
    env = {k: v for k, v in os.environ.items() if k not in ("TINYLLM_DEVICE", "DEV", "TINYLLM_PROGRAM_CACHE")}
    env |= {"JITBEAM": "2", "TINYLLM_STATIC_SIZES": "768"}
    cmd = [sys.executable, "tools/program_build_check.py", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=2400, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "0 failure(s)" in proc.stdout and "FAIL" not in proc.stdout


@pytest.mark.gpu
def test_flash_next_loads_its_programs_and_computes_the_same() -> None:
    """Qwen3.8-Flash-Next `UD-Q4_K_XL` with its MTP block and 256-token chunks (few programs): the process that loads
    the programs builds none and reproduces prompts, plain and sampled decoding, draft-and-verify cycles, a copied
    block and a resumed prompt bit for bit (`tools/program_cache_check.py`)."""
    from tinyllm.server.serve import find_mtp
    from tinyllm.weights.resolve import find_model

    try:
        path = find_model("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if find_mtp(path) is None:
        pytest.skip(f"no MTP block next to {path.name}")
    env = {k: v for k, v in os.environ.items() if k not in ("TINYLLM_DEVICE", "DEV", "TINYLLM_PROGRAM_CACHE")}
    env["JITBEAM"] = "2"
    cmd = [sys.executable, "tools/program_cache_check.py", str(path), "--chunk", "256"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=3000, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "0 failure(s)" in proc.stdout and "FAIL" not in proc.stdout
