#!/usr/bin/env python3
"""Decode attention on the matrix cores (tinyllm.tg.kernels.decode_mma): exact across row counts, accurate, fast.

    DEV=KFD+AMD uv run python tools/decode_mma_check.py [--rows 8] [--heads 24,4] [--context 8192] [--time 32768]

Random half-precision queries and cache. At positions around key-tile and split boundaries:
- every row of an M-row call must equal the 1-row call for that row, bit for bit (what keeps the speculative verify
  equal to decoding once both use this kernel);
- the result must be as close to a float64 reference as upstream's decode kernel is (both are reported).
With --time it times one row against upstream's decode kernel and M rows against the rows kernel the verify uses
now. Exits non-zero on a difference across row counts or an error well above upstream's.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import numpy as np


def reference(q: np.ndarray, cache: np.ndarray, valid_first: int) -> np.ndarray:
    """float64 attention: q (1, H, M, D), cache (2, 1, H_KV, N, D); row j over the first valid_first + j keys."""
    _, H, M, D = q.shape
    group = H // cache.shape[2]
    out = np.zeros(q.shape)
    for h in range(H):
        k, v = cache[0, 0, h // group].astype(np.float64), cache[1, 0, h // group].astype(np.float64)
        for j in range(M):
            n = valid_first + j
            s = k[:n] @ q[0, h, j].astype(np.float64) / np.sqrt(D)
            p = np.exp(s - s.max())
            out[0, h, j] = p @ v[:n] / p.sum()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--heads", default="24,4", help="query heads and KV heads (default: the 27B's)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--time", type=int, default=0, metavar="N", help="also time at N positions")
    ap.add_argument("--grid", default=None, help="range ids of the grid axes, e.g. head=0,split=1")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a decode_mma setting")
    args = ap.parse_args()

    from tinyllm.tg import Tensor, TinyJit, UOp, synchronize  # chooses the device before tinygrad is imported
    from tinyllm.tg.kernels import attention, decode_mma

    decode = importlib.import_module("tinygrad.llm.kernels.amd").amd_flash_attention_decode
    if args.grid:
        decode_mma.GRID = {k: int(x) for k, x in (kv.split("=") for kv in args.grid.split(","))}
    for setting in args.set:
        name, value = setting.split("=", 1)
        setattr(decode_mma, name, int(value))
    H, H_KV = (int(x) for x in args.heads.split(","))
    M, D = args.rows, 256
    rng = np.random.default_rng(0)
    failures = 0

    def setup(n: int) -> tuple[np.ndarray, np.ndarray, Tensor, Tensor, UOp]:
        cache_np = rng.standard_normal((2, 1, H_KV, n, D)).astype(np.float16)
        q_np = (rng.standard_normal((1, H, M, D)) * 2).astype(np.float16)
        return cache_np, q_np, Tensor(cache_np).realize(), Tensor(q_np).realize(), UOp.variable("pos", 0, n - 1)

    n = args.context
    cache_np, q_np, cache, q, v = setup(n)
    rows_jit = TinyJit(lambda q, c, p: decode_mma.flash_decode_mma(q, c, p + 1).realize())
    one_jit = TinyJit(lambda q, c, p: decode_mma.flash_decode_mma(q, c, p + 1).realize())
    upstream_jit = TinyJit(lambda q, c, p: decode(q, c, p + 1, n).realize())
    singles = [Tensor(q_np[:, :, j : j + 1].copy()).realize() for j in range(M)]
    positions = [0, 1, 30, 31, 32, 33, 100, 1500, 1530, 1535, 1536, 1537, 3000, 5000, n - M]
    worst_mma = worst_upstream = 0.0
    for p in sorted({p for p in positions if p <= n - M}):
        many = rows_jit(q, cache, v.bind(p)).numpy()
        ones = np.concatenate([one_jit(singles[j], cache, v.bind(p + j)).numpy() for j in range(M)], axis=2)
        ups = np.concatenate([upstream_jit(singles[j], cache, v.bind(p + j)).numpy() for j in range(M)], axis=2)
        ref = reference(q_np, cache_np, p + 1)
        same = np.array_equal(many, ones)
        failures += not same
        err_mma, err_up = float(np.abs(many - ref).max()), float(np.abs(ups - ref).max())
        worst_mma, worst_upstream = max(worst_mma, err_mma), max(worst_upstream, err_up)
        across = "same" if same else f"FAIL across row counts (max diff {np.abs(many - ones).max():.3g})"
        print(
            f"tinyllm decode mma, {M} rows at {p + 1}..{p + M}: {across}; "
            f"error {err_mma:.2e} (upstream's kernel {err_up:.2e})"
        )
    if worst_mma > 4 * worst_upstream + 1e-3:
        failures += 1
        print(f"FAIL: error {worst_mma:.2e} well above upstream's {worst_upstream:.2e}")
    if args.time:
        n = args.time
        _, q_np, cache, q, v = setup(n)
        q1 = Tensor(q_np[:, :, :1].copy()).realize()
        p = n - M
        runs = [
            ("upstream decode kernel, 1 row", TinyJit(lambda q, c, p: decode(q, c, p + 1, n).realize()), q1),
            ("matrix cores, 1 row", TinyJit(lambda q, c, p: decode_mma.flash_decode_mma(q, c, p + 1).realize()), q1),
            (
                f"rows kernel, {M} rows",
                TinyJit(lambda q, c, p: attention.flash_decode_rows(q, c, p + 1, n).realize()),
                q,
            ),
            (f"matrix cores, {M} rows", TinyJit(lambda q, c, p: decode_mma.flash_decode_mma(q, c, p + 1).realize()), q),
        ]
        for name, fn, x in runs:
            for _ in range(3):
                fn(x, cache, v.bind(p))
            synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn(x, cache, v.bind(p))
            synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1e3
            print(f"  {name}: {ms:.3f} ms at {n} positions ({2 * H_KV * n * D * 2 / ms / 1e6:.0f} GB/s of the cache)")
    print("tinyllm decode mma: " + ("all identical" if not failures else f"{failures} FAIL"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
