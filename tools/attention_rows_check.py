#!/usr/bin/env python3
"""Multi-row flash decode attention (tinyllm.tg.kernels.attention) against decoding's kernel run once per row.

    DEV=KFD+AMD uv run python tools/attention_rows_check.py [--rows 8] [--context 8192] [--time 32768]

Random half-precision queries and cache in the 27B's shapes (24 query heads, 4 KV heads, head size 256). At
positions around chunk boundaries and past 3,072 keys (where a block runs several rounds), the rows kernel must
equal `amd_flash_attention_decode` called for each row with that row's length, bit for bit. With --time it also
times both at that context. Exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--context", type=int, default=8192, help="cache positions for the equality checks")
    ap.add_argument("--time", type=int, default=0, metavar="N", help="also time both at N positions")
    ap.add_argument("--grid", default=None, help="order of the grid axes, e.g. row,head,chunk (fastest first)")
    ap.add_argument("--heads", default="24,4", help="query heads and KV heads (default: the 27B's)")
    args = ap.parse_args()

    import importlib

    from tinyllm.tg import Tensor, TinyJit, UOp, synchronize  # chooses the device before tinygrad is imported
    from tinyllm.tg.kernels import attention

    amd_flash_attention_decode = importlib.import_module("tinygrad.llm.kernels.amd").amd_flash_attention_decode

    if args.grid:
        attention.GRID = {name: i for i, name in enumerate(args.grid.split(","))}
    H, H_KV = (int(x) for x in args.heads.split(","))
    M, D = args.rows, 256
    rng = np.random.default_rng(0)
    failures = 0

    def build(n: int) -> tuple[object, object, object, object]:
        cache = Tensor(rng.standard_normal((2, 1, H_KV, n, D)).astype(np.float16)).realize()
        q = Tensor((rng.standard_normal((1, H, M, D)) * 2).astype(np.float16)).realize()
        v = UOp.variable("pos", 0, n - M - 1)

        def rows(q: Tensor, cache: Tensor, pos: UOp) -> Tensor:
            return attention.flash_decode_rows(q, cache, pos + 1, n).realize()

        def per_row(q: Tensor, cache: Tensor, pos: UOp) -> Tensor:
            outs = [amd_flash_attention_decode(q[:, :, j : j + 1], cache, pos + (j + 1), n) for j in range(M)]
            return outs[0].cat(*outs[1:], dim=2).realize()

        return cache, q, v, (TinyJit(rows), TinyJit(per_row))

    n = args.context
    cache, q, v, (rows_jit, ref_jit) = build(n)
    positions = [0, 1, 55, 56, 57, 63, 64, 100, 3000, 3064, 3065, 3071, 3072, 4000, 6100, n - M - 1]
    for p in [p for p in positions if p <= n - M - 1]:
        got = rows_jit(q, cache, v.bind(p)).numpy()
        ref = ref_jit(q, cache, v.bind(p)).numpy()
        same = np.array_equal(got, ref)
        failures += not same
        detail = "" if same else f" FAIL (max diff {np.abs(got - ref).max():.3g})"
        print(f"tinyllm rows attention, {M} rows at positions {p + 1}..{p + M}: {'same' if same else ''}{detail}")
    if args.time:
        n = args.time
        cache, q, v, (rows_jit, ref_jit) = build(n)
        p = n - M - 1
        for name, fn in (("one call per row", ref_jit), ("rows kernel", rows_jit)):
            for _ in range(3):
                fn(q, cache, v.bind(p))
            synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn(q, cache, v.bind(p))
            synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1e3
            moved = 2 * H_KV * n * D * 2 / 1e9
            rate = moved / ms * 1e3
            print(f"  {name}: {ms:.2f} ms for {M} rows at {n} positions ({rate:.0f} GB/s of the cache read once)")
    print("tinyllm rows attention: " + ("all identical" if not failures else f"{failures} FAIL"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
