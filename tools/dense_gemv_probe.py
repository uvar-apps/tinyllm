#!/usr/bin/env python3
"""Experiment: does tinygrad's own dense fp16 matrix-vector kernel fix large BF16/F16 layers?

At the pinned commit `Linear` uses `f16_gemv` only for float matrices with at most 2048 outputs.
Larger ones, such as the BF16 output head (248,320 x 5,120) of Qwen3.8-27B-UD-Q8_K_XL, fall to
tinygrad's generic code generation, which reads them at about 9 GB/s on Strix Halo.

This script raises that limit in this process only (the installed tinygrad is not modified) and
measures single-row decode with and without the change.

    uv run python tools/dense_gemv_probe.py <model.gguf> [--limit N] [--rows 1]

The kernel computes each (token, output) pair in its own workgroup, so it re-reads the weight
matrix once per row: a fix for one-row decode, not for batches.
"""

from __future__ import annotations

import argparse
import inspect
import statistics
import textwrap
import time

ORIGINAL = "self.out_features <= 2048"


def raise_limit(limit: int) -> None:
    from tinygrad.llm.kernels import amd

    source = textwrap.dedent(inspect.getsource(amd.Linear.__call__))
    if source.count(ORIGINAL) != 1:
        raise SystemExit(f"tinygrad changed: {ORIGINAL!r} not found once in Linear.__call__")
    namespace: dict[str, object] = {}
    exec(compile(source.replace(ORIGINAL, f"self.out_features <= {limit}"), amd.__file__, "exec"), vars(amd), namespace)
    amd.Linear.__call__ = namespace["__call__"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=0, help="new output limit; 0 keeps upstream's 2048")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    from tinyllm import device

    device.configure("amd")
    from tinyllm.tg.upstream import UpstreamSession
    from tinyllm.weights.resolve import find_model

    if args.limit:
        raise_limit(args.limit)
    session = UpstreamSession(find_model(args.model), 2048, rows=args.rows, chunk=32)
    session.prefill([1000 + i for i in range(64)])
    start = time.perf_counter()
    tokens = []
    for _ in range(args.steps):
        tokens.append(int(session.step()[0]))
    decode = [r for r in session.trace if r.kind == "decode"][3:]
    ms = statistics.median(r.seconds for r in decode) * 1e3
    gb = statistics.median(r.bytes_touched for r in decode) / 1e9
    label = f"limit {args.limit}" if args.limit else "upstream limit 2048"
    print(
        f"{label}: {args.rows} row(s), {ms:.1f} ms/step, {args.rows * 1e3 / ms:.2f} tok/s, "
        f"{gb:.1f} GB touched per step, wall {time.perf_counter() - start:.1f} s for {args.steps} steps"
    )
    print(f"tokens {tokens}")


if __name__ == "__main__":
    main()
