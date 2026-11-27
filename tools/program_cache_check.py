#!/usr/bin/env python3
"""A session whose programs were loaded from disk against one that built them (tinyllm.tg.programs).

    DEV=KFD+AMD JITBEAM=2 uv run python tools/program_cache_check.py <model.gguf> [--mtp mtp.gguf] [--chunk 256]
    DEV=KFD+AMD uv run python tools/program_cache_check.py Qwen3.8-27B-UD-Q4_K_XL.gguf --chunk 128 \
        [--draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf -k 7]
    uv run python tools/program_cache_check.py Qwen3.5-0.8B-Q8_0.gguf --device cpu --ctx 512 --chunk 16 --prompt 40

Runs itself twice as a child process on an empty program directory. The first child builds the session's programs,
saves them and runs a fixed workload; the second must find them, build none, and produce the same numbers bit for
bit: the logits after a prompt, plain decoding, and where the model drafts, the sampler's results of draft-and-verify
cycles, a copied block's verify, the cycles after it (a rollback) and decoding with the sampler on the GPU; then a
second prompt that changes the first one's last tokens, which resumes from a state checkpoint.

Both children start from the model's own state at construction, so equal outputs mean the loaded programs read
and write the buffers the built ones do, and that the caches a loading process creates zeroed behave as the ones
the model made. Prints both children's times from process start to ready. Exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

T0 = time.perf_counter()
GREEDY, SAMPLED = [0.0, 1.0, 1.0, 0.0, 0.0], [1.0, 20, 0.95, 0.0, 0.5]


def child(args: argparse.Namespace) -> int:
    import numpy as np

    from tinyllm import device

    device.configure(args.device)
    from tinyllm.runner.upstream import UpstreamRunner
    from tinyllm.server.serve import find_mtp, has_mtp
    from tinyllm.tg import synchronize

    path = Path(args.model).expanduser()
    drafter = Path(args.draft_model).expanduser() if args.draft_model else None
    mtp = None
    if args.k and not drafter and not has_mtp(path):
        mtp = Path(args.mtp).expanduser() if args.mtp else find_mtp(path)
    k = args.k if drafter or mtp is not None or has_mtp(path) else 0
    runner = UpstreamRunner(
        path,
        args.ctx,
        chunk=args.chunk,
        speculative=k,
        drafter=drafter if k else None,
        mtp_path=mtp,
        copy=args.copy if mtp else 0,
        programs=args.dir,
    )
    programs = runner.session.programs
    runner.warmup()
    ready = time.perf_counter() - T0
    traced = [name for name, jit in programs.jits.items() if jit.cnt and name not in programs.loaded]
    print(f"ready after {ready:.1f} s: restored {programs.restored}, {len(programs.loaded)} programs loaded, "
          f"{len(traced)} built, {programs.bytes / 1e6:.1f} MB {programs.note}")  # fmt: skip
    print("stages: " + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in runner.warmup_stages.items()))
    if programs.pending():
        print(f"FAIL programs left half built by the warm-up: {programs.pending()}")
        return 1

    if args.text:
        text = Path(args.text).read_text()
    else:  # code, as the model is used: this repository's own sources
        text = "\n".join(p.read_text() for p in sorted((Path(__file__).parent.parent / "src").rglob("*.py"))[:40])
    ids = runner.session.tokenizer().encode(text)
    n, new = args.prompt, args.new
    if len(ids) < 2 * n + new + 200:
        raise SystemExit(f"the text has {len(ids)} tokens, {2 * n + new + 200} are needed")
    out: dict[str, np.ndarray] = {}

    def prompt(tokens: list[int], name: str) -> int:
        start = runner.begin(tokens)
        logits = None
        for lo in range(start, len(tokens), 8192):  # the engine's slices
            piece = tokens[lo : lo + 8192]
            end = lo + len(piece)
            logits = runner.feed(piece, tokens[end] if end < len(tokens) else None)
        synchronize()
        out[name], out[name + ", tokens reused"] = np.array(logits, copy=True), np.array([runner.reused])
        return int(np.argmax(logits))

    def cycles(tokens: list[int], count: int, name: str) -> None:
        if not runner.speculative:
            return
        seen = []
        for _ in range(count):
            rows, accepted = runner.feed_speculative(tokens[-1], [GREEDY] * runner.speculative_rows)
            seen.append(np.array(rows, copy=True))
            tokens += [int(rows[j, 0]) for j in range(accepted + 1)]
        out[name] = np.stack(seen)

    tokens = [prompt(ids[:n], "prompt")]
    plain = []
    for _ in range(args.steps):
        plain.append(np.array(runner.feed([tokens[-1]]), copy=True))
        tokens.append(int(np.argmax(plain[-1])))
    out["plain decoding"] = np.stack(plain)
    cycles(tokens, 3 * args.steps, "draft-and-verify cycles")
    if runner.copy:
        block = ids[n + 100 : n + 100 + runner.copy]
        rows, accepted = runner.feed_speculative(tokens[-1], [GREEDY] * runner.speculative_rows, block)
        out["copied block"] = np.array(rows, copy=True)
        tokens += [int(rows[j, 0]) for j in range(accepted + 1)]
        cycles(tokens, 4, "cycles after the copied block")
    if runner.device_sampling:
        out["sampled on the GPU"] = np.stack(
            [np.array(runner.feed_sampled(tokens[-1], SAMPLED), copy=True) for _ in range(2)]
        )
    out["tokens"] = np.array(tokens)
    # an agent's next step: the prompt again with its last tokens changed and new ones after them
    again = [prompt(ids[: n - min(40, n // 2)] + ids[2 * n : 2 * n + new], "second prompt")]
    cycles(again, 4, "cycles after the second prompt")
    out["tokens after the second prompt"] = np.array(again)
    np.savez(args.child, **out)  # type: ignore[arg-type]
    if programs.restored and (
        late := [name for name, jit in programs.jits.items() if jit.cnt and name not in programs.loaded]
    ):
        print(f"FAIL programs built although the session was loaded: {late}")
        return 1
    print(f"tokens {tokens[:16]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--mtp", default=None, help="the MTP block's GGUF (default: found as `tinyllm serve` does)")
    ap.add_argument("--draft-model", default=None, help="a DFlash2 drafter's GGUF: draft with it instead")
    ap.add_argument("--device", default=None, help="auto, amd, cpu (default: auto)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument(
        "--chunk", type=int, default=256, help="prefill chunk: 256 builds few programs (default: %(default)s)"
    )
    ap.add_argument("-k", type=int, default=4, help="drafts per cycle where the model has an MTP block")
    ap.add_argument("--copy", type=int, default=15, help="drafts of a copied block")
    ap.add_argument("--prompt", type=int, default=1200, help="prompt tokens")
    ap.add_argument("--new", type=int, default=500, help="tokens the second prompt adds")
    ap.add_argument("--steps", type=int, default=8, help="plain decoding steps")
    ap.add_argument("--text", default=None, help="a text file for the prompts (default: this repository's sources)")
    ap.add_argument("--dir", default=None, help="the program directory (default: a temporary one, removed after)")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    if args.child:
        return child(args)

    import numpy as np

    failures = 0
    with tempfile.TemporaryDirectory(prefix="tinyllm-programs-") as scratch:
        directory = Path(args.dir).expanduser() if args.dir else Path(scratch) / "programs"
        if directory.exists() and any(directory.iterdir()):
            raise SystemExit(f"{directory} is not empty: the first run must build the programs")
        outputs, reports = [], []
        for run in ("built", "loaded"):
            result = Path(scratch) / f"{run}.npz"
            cmd = [sys.executable, __file__, *sys.argv[1:], "--dir", str(directory), "--child", str(result)]
            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            lines = [
                line for line in proc.stdout.splitlines() if line.startswith(("ready", "stages", "FAIL", "tokens"))
            ]
            print(f"--- programs {run}: {time.perf_counter() - start:.0f} s in all")
            print("\n".join(lines))
            if proc.returncode != 0 or not result.exists():
                print(f"FAIL the run with programs {run} exited with {proc.returncode}")
                print(proc.stdout[-2000:] + "\n" + proc.stderr[-3000:])
                return 1
            outputs.append(np.load(result))
            reports.append("\n".join(lines))
        for run, report, want in zip(("built", "loaded"), reports, ("restored False", "restored True, "), strict=True):
            if want not in report or (run == "loaded" and ", 0 built" not in report):
                print(f"FAIL the run with programs {run} did not report `{want}` and, when loaded, none built")
                failures += 1
        built, loaded = outputs
        for name in built.files:
            if name not in loaded.files or built[name].shape != loaded[name].shape:
                print(f"FAIL {name}: missing or of another shape in the run with loaded programs")
                failures += 1
            elif np.array_equal(built[name], loaded[name], equal_nan=True):
                print(f"ok   {name}: identical {built[name].shape}")
            else:
                d = np.abs(built[name].astype(np.float64) - loaded[name].astype(np.float64))
                print(f"FAIL {name}: differs by up to {np.nanmax(d):.4g} in {int((d > 0).sum())} of {d.size} numbers")
                failures += 1
    print(f"tinyllm program cache check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
