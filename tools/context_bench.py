#!/usr/bin/env python3
"""Decode speed against context length: plain one-token steps and speculative cycles at growing positions.

    DEV=KFD+AMD uv run python tools/context_bench.py ~/.cache/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
        [--drafter DFlash2.gguf] [--lengths 1024,8192,16384,32768]

One session is prefilled with Python and Rust code (tools/agent_bench.py's files) up to each length in turn. At
each length it times one-token decode steps and greedy speculative cycles (MTP, or DFlash2 with --drafter), and
reports milliseconds per step, tokens per cycle and tokens per second. The cycle time is what the context costs:
attention reads every cached key and value once per verify row. Tokens per cycle depend on what the model writes
there.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

GREEDY = [0.0, 1.0, 1.0, 0.0, 0.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--drafter", default=None, help="a DFlash2 GGUF (default: the model's MTP block)")
    ap.add_argument("--speculative", type=int, default=None, help="drafts per cycle (default: 3 MTP, 7 DFlash2)")
    ap.add_argument("--lengths", default="1024,8192,16384,32768")
    ap.add_argument("--steps", type=int, default=16, help="timed decode steps per length")
    ap.add_argument("--cycles", type=int, default=24, help="timed speculative cycles per length")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    sys.path.insert(0, str(Path(__file__).parent))
    import agent_bench

    from tinyllm.tg.upstream import UpstreamSession
    from tinyllm.weights.resolve import find_model

    lengths = [int(x) for x in args.lengths.split(",")]
    drafter = Path(args.drafter).expanduser() if args.drafter else None
    k = args.speculative if args.speculative is not None else (7 if drafter else 3)
    start = time.perf_counter()
    s = UpstreamSession(
        find_model(args.model), max(lengths) + 2048, chunk=128, logits=True, speculative=k, drafter=drafter
    )
    tok = s.tokenizer()
    text = "".join(f"// {p}\n{c}\n" for p, c in agent_bench.FILES.items())
    corpus = tok.encode(text)
    corpus = (corpus * (max(lengths) // len(corpus) + 2))[: max(lengths)]
    print(
        f"{Path(args.model).name}: {'DFlash2' if drafter else 'MTP'}, {k} drafts per cycle; "
        f"loaded in {time.perf_counter() - start:.0f} s"
    )
    print(f"{'context':>8}  {'decode ms':>9}  {'tok/s':>6}  {'cycle ms':>8}  {'tok/cycle':>9}  {'tok/s':>6}")
    rows = [GREEDY] * (k + 1)
    # compile every program before timing: both prefill programs, decode, the cycle and each rollback
    s.prefill(corpus[:300], wait_each=False)
    for _ in range(2):
        s.step([corpus[0]])
    for _ in range(3):
        s.speculate(corpus[0], rows)
    for _ in range(2):
        for n in range(k):
            s._restore_row = n
            s._settle()
    s.reset()
    for length in lengths:
        if s.pos < length:
            s.prefill(corpus[s.pos : length], wait_each=False)
        token = int(np.asarray(s.step([corpus[length - 1]])[0]).argmax())  # also compiles the decode program
        times = []
        for i in range(args.steps + 3):
            t0 = time.perf_counter()
            token = int(np.asarray(s.step([token])[0]).argmax())
            if i >= 3:
                times.append(time.perf_counter() - t0)
        decode = float(np.median(times))
        cycle_times, produced = [], []
        for i in range(args.cycles + 3):
            t0 = time.perf_counter()
            out, n = s.speculate(token, rows)
            dt = time.perf_counter() - t0
            token = int(out[n, 0])
            if i >= 3:
                cycle_times.append(dt)
                produced.append(n + 1)
        cycle = float(np.median(cycle_times))
        per = float(np.mean(produced))
        print(
            f"{length:8d}  {decode * 1e3:9.1f}  {1 / decode:6.1f}  {cycle * 1e3:8.1f}  {per:9.2f}  "
            f"{sum(produced) / sum(cycle_times):6.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
