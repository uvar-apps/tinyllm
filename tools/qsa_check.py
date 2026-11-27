#!/usr/bin/env python3
"""Qwen3.8-Flash-Next's sparse attention kernels (tinyllm.tg.kernels.qsa) against float64 references.

    DEV=KFD+AMD uv run python tools/qsa_check.py [--context 8192] [--time 16000] [--set WAYS=2 --set SEARCH_PASSES=31]

Random raw indexer keys, queries and cache at Flash-Next's shapes (4 indexer heads of 128, blocks of 4, top 512;
24 query heads, 2 KV heads of 256). For decode rows at positions around the 2,051-position bound and for prefill
chunks across it:
- the block keys must match a float64 mean, RMSNorm and rotary;
- the scores must match a float64 product of the kernel's own block keys;
- each list must be exactly the top 512 of the kernel's own scores (ties lowest block first), ascending, then the
  tail block;
- the attention must match float64 attention over the listed positions as closely as decoding's kernel matches
  dense attention.
With --time it times decode and a prefill call (--time-rows, 128) at N positions. Exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import numpy as np

KPOOL, TOPB, IH, ID, H, H_KV, D = 4, 512, 4, 128, 24, 2, 256


def ref_block_keys(kg: np.ndarray, w: np.ndarray, freqs: np.ndarray, eps: float) -> np.ndarray:
    """(blocks, 128) float64 from raw keys (positions, 128)."""
    x = kg.astype(np.float64).reshape(-1, KPOOL, ID).mean(1)
    x = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
    f = freqs[: x.shape[0] * KPOOL : KPOOL].astype(np.float64)
    half = f.shape[1] // 2
    cos, sin = f[:, :half], f[:, half:]
    x1, x2 = x[:, :half].copy(), x[:, half : 2 * half].copy()
    x[:, :half], x[:, half : 2 * half] = x1 * cos - x2 * sin, x2 * cos + x1 * sin
    return x


def ref_list(scores: np.ndarray, q_pos: int) -> list[int]:
    n = (q_pos + 1) // KPOOL
    if n <= TOPB:
        picked = list(range(n))
    else:
        s = scores[:n]
        order = np.lexsort((np.arange(n), -s))  # descending score, then lowest block
        picked = sorted(order[:TOPB].tolist())
    return picked + ([n] if (q_pos + 1) % KPOOL else [])


def ref_attention(q: np.ndarray, cache: np.ndarray, blocks: list[int], q_pos: int) -> np.ndarray:
    """(H, D) float64 for one row's query (H, D) over the listed positions of cache (2, H_KV, N, D)."""
    pos = np.array([b * KPOOL + j for b in blocks for j in range(KPOOL) if b * KPOOL + j <= q_pos])
    out = np.zeros((H, D))
    for h in range(H):
        k, v = cache[0, h // (H // H_KV), pos].astype(np.float64), cache[1, h // (H // H_KV), pos].astype(np.float64)
        s = k @ q[h].astype(np.float64) / np.sqrt(D)
        p = np.exp(s - s.max())
        out[h] = p @ v / p.sum()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--time", type=int, default=0, metavar="N", help="also time at N positions")
    ap.add_argument("--time-rows", type=int, default=128, help="rows of the timed prefill call (default: %(default)s)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a qsa setting, e.g. WAYS=2")
    args = ap.parse_args()

    from tinyllm.tg import Tensor, UOp, dtypes, synchronize  # chooses the device before tinygrad is imported
    from tinyllm.tg.kernels import qsa

    for setting in args.set:
        name, value = setting.split("=", 1)
        setattr(qsa, name, int(value))

    precompute_freqs_cis = importlib.import_module("tinygrad.llm.model").precompute_freqs_cis

    N, R, eps = args.context, 128, 1e-6
    rng = np.random.default_rng(0)
    kg_np = rng.standard_normal((1, N, ID)).astype(np.float16)
    w_np = (0.5 + rng.random(ID)).astype(np.float32)
    freqs = precompute_freqs_cis(2 * 32, N + R, 1e7).realize()
    freqs_np = freqs.numpy()
    cache_np = rng.standard_normal((2, 1, H_KV, N, D)).astype(np.float16)
    kg, w, cache = Tensor(kg_np).realize(), Tensor(w_np).realize(), Tensor(cache_np).realize()
    blk = Tensor.zeros(1, ID, N // KPOOL, dtype=dtypes.float32).contiguous().realize()
    v_pos, v_rows = UOp.variable("start_pos", 0, N - 1), UOp.variable("toks", 1, R)
    failed = False

    # block keys: all positions in one call, then a few in the decode form, which must give the same keys
    blk = qsa.update_block_keys(blk, kg, w, freqs, v_pos.bind(0), N, N, eps, KPOOL)
    blk = blk.realize()
    got = blk.numpy()[0].T
    want = ref_block_keys(kg_np[0], w_np, freqs_np, eps)
    err = np.abs(got - want).max() / np.abs(want).max()
    print(f"block keys: max error {err:.2e} of the largest")
    failed |= err > 1e-5
    before = got.copy()
    for p in (0, 7, 2050, N - 1):
        blk = qsa.update_block_keys(blk, kg, w, freqs, v_pos.bind(p), 1, 1, eps, KPOOL).realize()
    same = np.array_equal(blk.numpy()[0].T, before)
    print(f"block keys again one position at a time: {'identical' if same else 'DIFFERENT'}")
    failed |= not same

    def run(q_first: int, rows: int, symbolic: bool) -> bool:
        pad = R if symbolic else rows
        iq_np = rng.standard_normal((1, pad, IH, ID)).astype(np.float32)
        q_np = rng.standard_normal((1, H, pad, D)).astype(np.float32)
        iq, q = Tensor(iq_np).realize(), Tensor(q_np).realize()
        pos, n_rows = v_pos.bind(q_first), (v_rows.bind(rows) if symbolic else rows)
        scores = Tensor.empty(1, pad, N // KPOOL, dtype=dtypes.float32)
        fxn = qsa.functools.partial(qsa._scores, q_first=pos, rows=n_rows, kpool=KPOOL, topb=TOPB)
        scores = Tensor.custom_kernel(scores, qsa._bound(iq, pos, n_rows), blk, fxn=fxn)[0]
        sel = Tensor.empty(1, pad, TOPB + 1, dtype=dtypes.int32)
        fxn = qsa.functools.partial(qsa._select, q_first=pos, rows=n_rows, kpool=KPOOL, topb=TOPB)
        sel = Tensor.custom_kernel(sel, qsa._bound(scores, pos, n_rows), fxn=fxn)[0]
        out = qsa.attend(q, cache, sel, pos, n_rows, KPOOL, TOPB)
        Tensor.realize(scores, sel, out)
        scores_np, sel_np, out_np = scores.numpy()[0], sel.numpy()[0], out.numpy()[0]
        keys = blk.numpy()[0].T.astype(np.float64)
        ok, worst_s, worst_a, sparse_rows = True, 0.0, 0.0, 0
        for r in range(rows):
            q_pos = q_first + r
            n = (q_pos + 1) // KPOOL
            if n > TOPB:
                sparse_rows += 1
                ref_s = np.maximum(iq_np[0, r].astype(np.float64) @ keys[:n].T, 0).sum(0) / np.sqrt(ID)
                worst_s = np.maximum(worst_s, np.abs(scores_np[r, :n] - ref_s).max() / np.abs(ref_s).max())
            blocks = ref_list(scores_np[r], q_pos)
            if sel_np[r, : len(blocks)].tolist() != blocks:
                bad = next(i for i, (a, b) in enumerate(zip(sel_np[r].tolist(), blocks, strict=False)) if a != b)
                print(f"  row {r} (position {q_pos}): list differs at entry {bad}: {sel_np[r, bad]} vs {blocks[bad]}")
                ok = False
                continue
            want = ref_attention(q_np[0, :, r], cache_np[:, 0], blocks, q_pos)
            worst_a = np.maximum(worst_a, np.abs(out_np[:, r] - want).max() / np.abs(want).max())
        ok &= worst_s < 1e-5 and worst_a < 5e-3 and not np.isnan(out_np[:, :rows]).any()  # max() drops NaN
        form = f"{rows} row(s){' (symbolic)' if symbolic else ''} from {q_first}"
        print(f"{form}: {sparse_rows} sparse; scores {worst_s:.1e}, attention {worst_a:.1e}: {'ok' if ok else 'FAIL'}")
        return ok

    def run_lists(q_first: int, rows: int) -> bool:
        """Lists made by hand: even rows take the first 512 blocks, odd rows the last 512, so the first tiles of a
        group's union hold no key of its odd rows (the grouped kernel must not turn their rescaling into NaN)."""
        q_np = rng.standard_normal((1, H, rows, D)).astype(np.float32)
        sel_np = np.zeros((1, rows, TOPB + 1), dtype=np.int32)
        lists = []
        for r in range(rows):
            q_pos = q_first + r
            n = (q_pos + 1) // KPOOL
            assert n >= 2 * TOPB
            blocks = list(range(0, TOPB) if r % 2 == 0 else range(n - TOPB, n))
            sel_np[0, r, :TOPB] = blocks
            sel_np[0, r, TOPB] = n
            lists.append(blocks + ([n] if (q_pos + 1) % KPOOL else []))
        sel = Tensor(sel_np).realize()
        out_np = qsa.attend(Tensor(q_np).realize(), cache, sel, v_pos.bind(q_first), rows, KPOOL, TOPB).numpy()[0]
        worst = 0.0
        for r in range(rows):
            want = ref_attention(q_np[0, :, r], cache_np[:, 0], lists[r], q_first + r)
            worst = np.maximum(worst, float(np.abs(out_np[:, r] - want).max() / np.abs(want).max()))
        ok = bool(worst < 5e-3) and not np.isnan(out_np[:, :rows]).any()  # max() keeps 0.0 over a NaN
        print(
            f"{rows} rows from {q_first}, disjoint lists in each group: attention {worst:.1e}: {'ok' if ok else 'FAIL'}"
        )
        return ok

    failed |= not run_lists(N - 200, 128)
    for q_first in (0, 5, 2046, 2047, 2048, 2050, 2051, 2052, 3001, N - 2):
        failed |= not run(q_first, 1, False)
    for q_first, rows in ((1990, 128), (4000, 100), (N - 129, 128)):
        failed |= not run(q_first, rows, True)
    failed |= not run(2500, 128, False)

    if args.time:
        P = args.time
        assert P < N, "--time needs a larger --context"
        from tinyllm.tg import TinyJit

        for rows in (1, args.time_rows):
            iq = Tensor(rng.standard_normal((1, rows, IH, ID)).astype(np.float32)).realize()
            q = Tensor(rng.standard_normal((1, H, rows, D)).astype(np.float32)).realize()

            @TinyJit
            def step(iq: Tensor, q: Tensor, pos: UOp, rows: int = rows) -> Tensor:
                sel = qsa.select(iq, blk, pos, rows, KPOOL, TOPB)
                return qsa.attend(q, cache, sel, pos, rows, KPOOL, TOPB).realize()

            for _ in range(3):
                step(iq, q, v_pos.bind(P - rows))
            synchronize()
            t0, n = time.perf_counter(), 20
            for _ in range(n):
                step(iq, q, v_pos.bind(P - rows))
            synchronize()
            print(f"{rows} row(s) at {P} positions: {(time.perf_counter() - t0) / n * 1e3:.3f} ms a layer")
    print(f"tinyllm qsa check: {'FAIL' if failed else 'all ok'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
