"""tinyllm's kernels: detection, layout and routing on the CPU, the kernels themselves on the GPU."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from conftest import REPO

from tinyllm import tg


def _decoded(bits: np.ndarray, ggml_type: int, shape: tuple[int, int]):  # type: ignore[no-untyped-def]
    """A weight wrapped exactly as tinygrad's GGUF loader wraps one read from a file."""
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor

    raw = Tensor(bits.reshape(-1).view(np.uint8)).realize()
    return ggml_data_to_tensor(raw, bits.size, ggml_type).reshape(shape).cast("float16")


def test_find_dense_recognises_bf16_and_f16_and_nothing_else() -> None:
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import find_dense

    bits = np.arange(64 * 32, dtype=np.uint16).reshape(64, 32)
    for ggml_type in (30, 1):
        found = find_dense(_decoded(bits, ggml_type, (64, 32)))
        assert found is not None and found[0] == ggml_type
        words = found[1].numpy()
        assert words.dtype == np.uint32 and words.shape == (64 * 32 // 2,)
        assert words[0] == (1 << 16) | 0  # weights 0 and 1, low half first
    # arithmetic after the decode, a quantized tensor, a tensor not read from a file
    assert find_dense(_decoded(bits, 30, (64, 32)) * 2) is None
    q8 = Tensor(np.zeros(34 * 4, dtype=np.uint8)).realize()
    assert find_dense(ggml_data_to_tensor(q8, 128, 8).reshape(4, 32)) is None
    assert find_dense(Tensor.ones(64, 32).contiguous().realize()) is None


@pytest.mark.parametrize(
    ("out_features", "tokens", "expected"),
    [
        (248320, 1, (1, 8, 4)),
        (248320, 8, (8, 8, 4)),
        (5120, 32, (8, 8, 4)),  # 32 rows run as four tiles of 8
        (48, 3, (3, 8, 2)),
        (96, 6, (6, 8, 4)),
        (100, 12, (6, 4, 1)),  # 100 rows are not a multiple of 8 waves; 12 tokens split into two tiles of 6
    ],
)
def test_layout(out_features: int, tokens: int, expected: tuple[int, int, int]) -> None:
    from tinyllm.tg.kernels.dense import _layout

    tile, waves, rows = _layout(out_features, tokens)
    assert (tile, waves, rows) == expected
    assert tokens % tile == 0 and out_features % (waves * rows) == 0 and 1024 % (waves * 32) == 0


def test_hook_is_idempotent_and_leaves_cpu_layers_alone() -> None:
    from tinygrad.llm.kernels.amd import Linear

    from tinyllm.tg import Tensor, kernels

    kernels.install()
    hooked = Linear.__call__
    kernels.install()
    assert Linear.__call__ is hooked and kernels.installed()

    bits = (np.arange(16 * 64, dtype=np.uint32).astype(np.uint16) | 0x3C00).reshape(16, 64)  # finite bf16 values
    layer = Linear(64, 16, bias=False)
    layer.weight = _decoded(bits, 30, (16, 64))
    x = np.random.default_rng(0).standard_normal((2, 64)).astype(np.float32)
    got = layer(Tensor(x)).numpy()
    # on the CPU the layer is not eligible and keeps tinygrad's path (which rounds bf16 to fp16 first)
    assert layer._tinyllm_dense is None
    weights = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(got, x @ weights.T, rtol=1e-3, atol=1e-3)


def _q4_decoded(ggml_type: int, rows: int, cols: int):  # type: ignore[no-untyped-def]
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor

    block = 18 if ggml_type == 2 else 20
    raw = Tensor(np.arange(rows * cols // 32 * block, dtype=np.uint32).astype(np.uint8)).realize()
    return ggml_data_to_tensor(raw, rows * cols, ggml_type).reshape(rows, cols).cast("float16")


def test_find_q4_recognises_q4_0_and_q4_1_only() -> None:
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import find_dense, find_q4

    for ggml_type, block in ((2, 18), (3, 20)):
        found = find_q4(_q4_decoded(ggml_type, 8, 64), 64)
        assert found is not None and found[0] == ggml_type
        assert found[1].dtype == tg.dtypes.uint16 and found[1].shape == (8 * 64 // 32 * block // 2,)
        assert find_dense(_q4_decoded(ggml_type, 8, 64)) is None
    assert find_q4(_q4_decoded(2, 8, 64) + 1, 64) is None
    assert find_q4(_q4_decoded(2, 8, 64), 48) is None  # rows must be whole blocks
    q8 = Tensor(np.zeros(34 * 4, dtype=np.uint8)).realize()
    assert find_q4(ggml_data_to_tensor(q8, 128, 8).reshape(4, 32), 32) is None


def test_prefill_finders_recognise_their_own_types_only() -> None:
    from types import SimpleNamespace

    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import iq4xs, q6k, q8, q45k

    finders = {"q45k": q45k.find_q45k, "iq4xs": iq4xs.find_iq4xs, "q6k": q6k.find_q6k, "q8": q8.find_q8}
    # ggml type: (bytes per 256 weights, the finder that serves it)
    cases = {12: (144, "q45k"), 13: (176, "q45k"), 23: (136, "iq4xs"), 14: (210, "q6k"), 8: (272, "q8")}
    for ggml_type, (size, owner) in cases.items():
        raw = Tensor(np.zeros(2 * size, dtype=np.uint8)).realize()
        weight = ggml_data_to_tensor(raw, 2 * 256, ggml_type).reshape(2, 256).cast("float16")
        layer = SimpleNamespace(weight=weight, in_features=256)
        for name, find in finders.items():
            found = find(layer)
            assert (found is not None) == (name == owner), (ggml_type, name)
            if found is not None:
                assert found[0] == ggml_type


def test_dense_mm_serves_multiples_of_16_rows_of_large_layers() -> None:
    from types import SimpleNamespace

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import densemm

    layer = SimpleNamespace(out_features=5120)
    for rows, served in ((1, False), (8, False), (12, False), (16, True), (24, False), (32, True), (128, True)):
        assert densemm.serves(layer, Tensor.empty(rows, 64)) == served, rows
    assert densemm.serves(SimpleNamespace(out_features=40), Tensor.empty(32, 64)) is False
    assert densemm.serves(SimpleNamespace(out_features=48), Tensor.empty(32, 64)) is False  # DeltaNet gates stay
    assert densemm.serves(layer, Tensor.empty(2, 16, 64))  # rows are counted over all leading dimensions
    assert densemm._config(32, 5120, 30)[1] == "bf16" and densemm._config(32, 5120, 1)[1] == "f16"
    assert densemm._config(32, 96, 30)[0] == 1  # 3 groups of 32 are not a whole number of 2-group steps


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("dense", "dense_mm", "q4", "q45k", "iq4xs", "q6k", "q8")),
        ("all", ("dense", "dense_mm", "q4", "q45k", "iq4xs", "q6k", "q8")),
        ("none", ()),
        ("0", ()),
        ("q4", ("q4",)),
        (" dense , q4 ", ("dense", "q4")),
    ],
)
def test_kernel_selection_from_the_environment(value: str | None, expected: tuple[str, ...]) -> None:
    from tinyllm.tg.kernels import ENV, requested

    assert requested({} if value is None else {ENV: value}) == expected


def test_unknown_kernel_names_are_rejected() -> None:
    from tinyllm.tg.kernels import ENV, install, requested

    with pytest.raises(ValueError, match="unknown kernel"):
        requested({ENV: "q5"})
    with pytest.raises(ValueError, match="unknown kernel"):
        install(["fast"])


@pytest.mark.gpu
@pytest.mark.parametrize(
    "check",
    [
        ["tools/dense_kernel_check.py"],
        ["tools/dense_kernel_check.py", "--f16"],
        ["tools/q4_kernel_check.py", "--rows", "1,3,8,16,32"],
        ["tools/q4_kernel_check.py", "--type", "q4_1", "--rows", "1,3,8,16,32"],
        ["tools/prefill_kernel_check.py", "--type", "q6_k", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q6_k", "--rows", "16,32,64,128", "--groups", "8", "--no-packed"],
        ["tools/prefill_kernel_check.py", "--type", "q8_0", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q8_0", "--rows", "16,32,64,128", "--groups", "8", "--no-packed"],
        ["tools/prefill_kernel_check.py", "--type", "q4_k", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q4_k", "--rows", "16,128", "--configs", "1u,2p,8p,8u"],
        ["tools/prefill_kernel_check.py", "--type", "q5_k", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q5_k", "--rows", "16,128", "--configs", "1u,2p,8p,8u"],
        ["tools/prefill_kernel_check.py", "--type", "iq4_xs", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "iq4_xs", "--rows", "16,128", "--configs", "1u,2p,8p,8u"],
        ["tools/prefill_kernel_check.py", "--type", "bf16", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "bf16", "--rows", "16,128", "--configs", "1b,2h,8b,8h"],
        ["tools/prefill_kernel_check.py", "--type", "f16", "--rows", "1,8,16,32,64"],
        [
            "tools/prefill_kernel_check.py",
            *("--type", "q5_k", "--rows", "64,128", "--configs", "1u,1p"),
            *("--set", "q45k.SHARED=(64,2,2)", "--set", "q45k.PREFETCH=True"),
        ],
        ["tools/sampler_check.py"],
    ],
    ids=[
        "bf16",
        "f16",
        "q4_0",
        "q4_1",
        "q6_k",
        "q6_k-unpacked",
        "q8_0",
        "q8_0-unpacked",
        "q4_k",
        "q4_k-configs",
        "q5_k",
        "q5_k-configs",
        "iq4_xs",
        "iq4_xs-configs",
        "bf16-prefill",
        "bf16-prefill-configs",
        "f16-prefill",
        "q5_k-shared-prefetch",
        "gpu-sampler",
    ],
)
def test_kernel_on_the_gpu(check: list[str]) -> None:
    """Small shapes through every code path of each kernel: decode and matrix-core, ragged chunks, odd row counts."""
    for extra in ([],):
        quick = [] if check[0].endswith("sampler_check.py") else ["--quick"]
        cmd = [sys.executable, check[0], *quick, *check[1:], *extra]
        if check[0].endswith("dense_kernel_check.py"):
            cmd += ["--rows", "1,3,8,32"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=600, env=_gpu_env())
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
        assert ("tinyllm" in proc.stdout or "draws match" in proc.stdout) and "FAIL" not in proc.stdout


def _gpu_env() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if k not in ("TINYLLM_DEVICE", "DEV")}
