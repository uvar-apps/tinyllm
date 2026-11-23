#!/usr/bin/env python3
"""Correctness and speed of tinyllm's BF16/F16 decode kernel on the GPU, on Qwen3.8-27B layer shapes.

    uv run python tools/dense_kernel_check.py                 # all shapes, 1 and 8 rows
    uv run python tools/dense_kernel_check.py --quick         # small shapes only, every code path
    uv run python tools/dense_kernel_check.py --baseline      # tinygrad's own path, for comparison

Weights are random BF16 (or F16) bytes wrapped exactly as tinygrad's GGUF loader wraps a tensor read
from a file (`ggml_data_to_tensor`, then a cast to fp16), so the layer goes through the same detection
as in a real model. Outputs are checked against NumPy in float64 on a sample of rows. Speed is the
median of warm JIT replays; bandwidth counts the weight bytes only. Exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

# (name, out_features, in_features)
SHAPES = [
    ("output head", 248320, 5120),
    ("ffn gate/up", 17408, 5120),
    ("ffn down", 5120, 17408),
    ("attn q", 12288, 5120),
    ("deltanet qkv", 10240, 5120),
    ("deltanet out", 5120, 6144),
    ("deltanet alpha", 48, 5120),
]
QUICK = [("small", 512, 1024), ("ragged chunk", 96, 1312), ("deltanet alpha", 48, 5120)]
SAMPLE_ROWS = 4096


def random_weights(rng: np.random.Generator, out: int, inp: int, half: bool) -> np.ndarray:
    """uint16 bit patterns of finite values between about 1e-5 and 0.5 in magnitude."""
    n = out * inp
    sign = rng.integers(0, 2, n, dtype=np.uint16) << 15
    if half:  # fp16: 5-bit exponent, 10-bit mantissa
        return (
            sign | (rng.integers(0, 15, n, dtype=np.uint16) << 10) | rng.integers(0, 1024, n, dtype=np.uint16)
        ).reshape(out, inp)
    return (
        sign | (rng.integers(110, 126, n, dtype=np.uint16) << 7) | rng.integers(0, 128, n, dtype=np.uint16)
    ).reshape(out, inp)


def to_float(bits: np.ndarray, half: bool) -> np.ndarray:
    return (
        bits.view(np.float16).astype(np.float64)
        if half
        else (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="do not install tinyllm's kernel")
    ap.add_argument("--rows", default="1,8")
    ap.add_argument("--f16", action="store_true", help="F16 weights instead of BF16")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from tinyllm import device

    device.configure("amd")
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.kernels.amd import Linear

    from tinyllm import tg
    from tinyllm.tg import kernels

    tg.set_polling_wait()
    if not args.baseline:
        kernels.install(["dense"])  # the decode kernel at every row count; dense_mm has tools/prefill_kernel_check.py

    rng = np.random.default_rng(0)
    rows_list = [int(r) for r in args.rows.split(",")]
    shapes = QUICK if args.quick else SHAPES
    ggml_type = 1 if args.f16 else 30
    failed = False
    print(f"{'layer':16s} {'out x in':>16s} {'rows':>4s} {'path':>9s} {'ms':>8s} {'GB/s':>6s} {'max rel err':>11s}")
    for name, out_f, in_f in shapes:
        bits = random_weights(rng, out_f, in_f, args.f16)
        raw = tg.Tensor(bits.reshape(-1).view(np.uint8)).realize()
        layer = Linear(in_f, out_f, bias=False)
        layer.weight = ggml_data_to_tensor(raw, out_f * in_f, ggml_type).reshape(out_f, in_f).cast("float16")
        sample = np.unique(np.concatenate([np.arange(min(out_f, 64)), rng.integers(0, out_f, min(out_f, SAMPLE_ROWS))]))
        w_sample = to_float(bits[sample], args.f16)
        for rows in rows_list:
            x_np = rng.standard_normal((rows, in_f)).astype(np.float32)
            x = tg.Tensor(x_np).realize()
            jit = tg.TinyJit(lambda t, layer=layer: layer(t).realize())
            got = None
            for _ in range(3):
                got = jit(x)
            times = []
            for _ in range(args.iters):
                start = time.perf_counter()
                got = jit(x)
                tg.synchronize()
                times.append(time.perf_counter() - start)
            y = got.numpy()[:, sample].astype(np.float64)
            ref = x_np.astype(np.float64) @ w_sample.T
            # fp16 weights in tinygrad's own path round BF16; compare relative to the row's scale
            scale = np.abs(x_np).astype(np.float64) @ np.abs(w_sample).T + 1e-30
            err = float(np.max(np.abs(y - ref) / scale))
            path = "tinyllm" if getattr(layer, "_tinyllm_dense", None) else "tinygrad"
            ms = statistics.median(times) * 1e3
            gbs = out_f * in_f * 2 / (ms / 1e3) / 1e9
            tol = 2e-6 if path == "tinyllm" else 2e-3
            bad = not err <= tol
            failed |= bad
            flag = "  FAIL" if bad else ""
            print(f"{name:16s} {f'{out_f}x{in_f}':>16s} {rows:4d} {path:>9s} {ms:8.3f} {gbs:6.0f} {err:11.2e}{flag}")
        del raw, layer
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
