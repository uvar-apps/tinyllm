#!/usr/bin/env python3
"""The GPU sampler (`tinyllm.tg.sampler.sample_row`) against a NumPy model of the same arithmetic.

    uv run python tools/sampler_check.py

For random rows of 248,320 logits and a range of settings and uniform numbers it checks the chosen token, the
32 candidates (exact top-k, ties to the lower id), and the log-sum-exp; then times one call. Exits non-zero on a
mismatch.
"""

from __future__ import annotations

import sys
import time

import numpy as np

VOCAB = 248_320


def reference(x: np.ndarray, params: list[float], k: int) -> tuple[int, np.ndarray, np.ndarray, float]:
    order = np.lexsort((np.arange(x.size), -x))[:k]
    vals = x[order].astype(np.float64)
    temperature, top_k, top_p, min_p, u = params
    p = np.exp((vals - vals[0]) / max(temperature, 1e-6)) * (np.arange(k) < top_k)
    p /= p.sum()
    keep = ((np.cumsum(p) - p) < top_p) & (p >= min_p * p[0])
    q = p * keep
    q /= q.sum()
    choice = min(int(np.sum(np.cumsum(q) <= u)), k - 1)
    lse = float(np.log(np.sum(np.exp(x.astype(np.float64) - x.max()))) + x.max())
    return int(order[choice]), vals, order, lse


def main() -> int:
    from tinyllm import device

    device.configure("amd")
    from tinygrad import Tensor, TinyJit

    from tinyllm import tg
    from tinyllm.tg.sampler import CANDIDATES, sample_row

    tg.set_polling_wait()
    rng = np.random.default_rng(0)
    settings = [
        [0.0, 1, 1.0, 0.0, 0.0],  # greedy
        [1.0, 20, 0.95, 0.0, None],  # Qwen3.8's defaults
        [0.7, 32, 0.9, 0.05, None],
        [1.5, 8, 1.0, 0.0, None],
        [1.0, 1, 1.0, 0.0, None],
    ]
    jit = TinyJit(lambda row, params: sample_row(row, params).realize())
    failed = checked = 0
    for case in range(6):
        x = (rng.standard_normal(VOCAB) * (2 + case)).astype(np.float32)
        row = Tensor(x).realize()
        for setting in settings:
            for _ in range(8):
                params = [float(rng.random()) if v is None else float(v) for v in setting]
                out = jit(row, Tensor(np.asarray(params, dtype=np.float32)).realize()).numpy()
                token, vals, ids, lse = reference(x, params, CANDIDATES)
                ok = int(out[0]) == token and np.array_equal(out[2 + CANDIDATES :].astype(np.int64), ids)
                ok = ok and np.allclose(out[2 : 2 + CANDIDATES], vals) and abs(float(out[1]) - lse) < 1e-3
                checked += 1
                if not ok:
                    failed += 1
                    print(f"MISMATCH case {case} params {params}: token {int(out[0])} vs {token}")
    params = Tensor(np.asarray([1.0, 20, 0.95, 0.0, 0.5], dtype=np.float32)).realize()
    for _ in range(3):
        jit(row, params)
    tg.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        jit(row, params)
    tg.synchronize()
    print(f"{checked - failed} of {checked} draws match; one call {(time.perf_counter() - start) / 50 * 1e3:.3f} ms")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
