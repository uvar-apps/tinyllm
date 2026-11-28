#!/usr/bin/env python3
"""A program captured by its first call against the same program built the ordinary way.

    DEV=KFD+AMD JITBEAM=2 TINYLLM_STATIC_SIZES=768 uv run python tools/program_build_check.py <...-00001-of-00004.gguf>

tinygrad builds a program in two calls: one that runs the graph as it is, one that captures it. Once a session has
built one program of a function, tinyllm captures the others of that function (its other lengths) on their first
call (`tinyllm.tg.programs.Program`), and lowers a program's tuned kernels in worker processes
(`tinyllm.tg.lowering`). Both only save time: the program must be the one two calls would have built.

This tool builds the full chunk's prefill program in two calls, then the 768-token program once by its first call
and once more in two calls, and compares the two at the same tokens from the same state: the logits and every
piece of recurrent state must be equal bit for bit, as must the first call's own results and the replay's.
Exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mtp", default=None, help="the MTP block's GGUF (default: found as `tinyllm serve` does)")
    ap.add_argument("--length", type=int, default=768, help="the fixed length to check (one of the session's)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from tinyllm.server.serve import find_mtp
    from tinyllm.tg import synchronize  # chooses the device, before anything imports tinygrad
    from tinyllm.tg.upstream import UpstreamSession

    model_path = Path(args.model).expanduser()
    mtp = Path(args.mtp).expanduser() if args.mtp else find_mtp(model_path)
    s = UpstreamSession(model_path, 8192, rows=1, chunk=2048, logits=True, speculative=4 if mtp else 0, mtp_path=mtp)
    length = args.length
    if length not in s._static or length == s.chunk:
        raise SystemExit(f"the session has no program of {length} tokens besides the chunk's ({sorted(s._static)})")
    text = "\n".join(p.read_text() for p in sorted((Path(__file__).parent.parent / "src").rglob("*.py"))[:40])
    ids = s.tokenizer().encode(text)[: s.chunk]
    if len(ids) < s.chunk:
        raise SystemExit(f"the text has {len(ids)} tokens, {s.chunk} are needed")
    program, whole = s._static[length], s._static[s.chunk]
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' if detail else ''}{detail}")

    def run(n: int) -> tuple[np.ndarray, list[np.ndarray], float]:
        """ids[:n] from position 0 in one call: the logits, the state after it, the seconds it took."""
        s.reset()
        t = time.perf_counter()
        out = s.prefill(ids[:n], wait_each=False)
        synchronize()
        seconds = time.perf_counter() - t
        return np.asarray(out[0]).copy(), [x.numpy().copy() for x in s._state_tensors()], seconds

    def same(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
        return bool(np.array_equal(a[0], b[0])) and all(np.array_equal(x, y) for x, y in zip(a[1], b[1], strict=True))

    took = [run(s.chunk)[2] for _ in range(2)]
    print(f"info the {s.chunk}-token program in two calls: {took[0]:.0f} s and {took[1]:.0f} s")
    check(
        f"the {s.chunk}-token program is captured, the {length}-token one not yet", whole.cnt >= 2 and program.cnt == 0
    )

    first = run(length)
    check(
        f"the {length}-token program was captured by its first call",
        program.early and not program.again and program.cnt == 2,
    )
    replay = run(length)
    print(f"info the {length}-token program by its first call: {first[2]:.0f} s, then {replay[2]:.2f} s a call")
    check("its first call's results are those of the captured program", same(first, replay))

    program.reset()  # tinygrad's: forget the capture, as if never called
    program.like, program.early = None, False
    took = [run(length)[2] for _ in range(2)]
    plain = run(length)
    print(f"info the {length}-token program in two calls: {took[0]:.0f} s and {took[1]:.0f} s")
    check("built in two calls it was not captured early", not program.early and program.cnt >= 2)
    worst = float(np.abs(plain[0] - replay[0]).max())
    check(
        "captured by its first call it computes what the program built in two calls computes",
        same(plain, replay),
        f"max diff {worst:.3g}",
    )
    print(f"tinyllm program build check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
