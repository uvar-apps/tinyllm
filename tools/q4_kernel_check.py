#!/usr/bin/env python3
"""Correctness and speed of tinyllm's Q4_0/Q4_1 kernels on the GPU, on Qwen3.8-27B layer shapes.

    uv run python tools/q4_kernel_check.py                    # Q4_0, all shapes, 1, 8 and 32 rows
    uv run python tools/q4_kernel_check.py --type q4_1
    uv run python tools/q4_kernel_check.py --quick            # small shapes, every code path
    uv run python tools/q4_kernel_check.py --baseline         # tinygrad's own path, for comparison

Blocks are random (fp16 scales, random 4-bit codes) and wrapped exactly as tinygrad's GGUF loader wraps a
tensor read from a file. The NumPy reference reproduces each path's arithmetic: on the decode path (row
counts that are not a multiple of 16) activations are quantized to int8 per group of 32; on the matrix-core
path (multiples of 16) weights and activations are rounded to fp16. Errors are relative to the size of each
dot product. Bandwidth counts the weight bytes only. Exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

SHAPES = [
    ("ffn gate/up", 17408, 5120),
    ("ffn down", 5120, 17408),
    ("attn q", 12288, 5120),
    ("deltanet qkv", 10240, 5120),
    ("deltanet out", 5120, 6144),
]
QUICK = [("small", 512, 1024), ("ragged chunk", 96, 1312), ("odd rows", 100, 2048)]
TYPES = {"q4_0": (2, 18), "q4_1": (3, 20)}
SAMPLE_ROWS = 2048


def random_blocks(rng: np.random.Generator, out: int, inp: int, kind: str) -> np.ndarray:
    fields = [("d", "<f2")] + ([("m", "<f2")] if kind == "q4_1" else []) + [("qs", "u1", (16,))]
    blocks = np.zeros(out * inp // 32, dtype=np.dtype(fields))
    blocks["d"] = rng.uniform(0.001, 0.01, blocks.size).astype(np.float16)
    if kind == "q4_1":
        blocks["m"] = rng.uniform(-0.05, 0.05, blocks.size).astype(np.float16)
    blocks["qs"] = rng.integers(0, 256, (blocks.size, 16), dtype=np.uint8)
    return blocks


def dequantize(blocks: np.ndarray, kind: str, rows: np.ndarray, inp: int) -> np.ndarray:
    per_row = inp // 32
    b = blocks.reshape(-1, per_row)[rows]
    q = np.concatenate([b["qs"] & 0x0F, b["qs"] >> 4], axis=-1).astype(np.float64)  # (rows, per_row, 32)
    d = b["d"].astype(np.float64)[..., None]
    w = d * (q - 8) if kind == "q4_0" else d * q + b["m"].astype(np.float64)[..., None]
    return w.reshape(len(rows), inp)


def quantize_activations(x: np.ndarray) -> np.ndarray:
    """tinygrad's q8_quantize: int8 per group of 32, scale max|x|/127, round half to even."""
    g = x.astype(np.float32).reshape(x.shape[0], -1, 32)
    scale = np.maximum(np.abs(g).max(-1, keepdims=True) / np.float32(127), np.float32(1e-8))
    return (np.clip(np.rint(g / scale), -127, 127) * scale).reshape(x.shape).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=sorted(TYPES), default="q4_0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="do not install tinyllm's kernels")
    ap.add_argument("--rows", default="1,8,32")
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
        kernels.install()

    ggml_type, block_bytes = TYPES[args.type]
    rng = np.random.default_rng(0)
    failed = False
    head = f"{'layer':16s} {'out x in':>12s} {'rows':>4s} {'path':>9s} {'ms':>8s} {'GB/s':>6s} {'max rel err':>11s}"
    print(f"{args.type}: {head}")
    for name, out_f, in_f in QUICK if args.quick else SHAPES:
        blocks = random_blocks(rng, out_f, in_f, args.type)
        raw = tg.Tensor(blocks.view(np.uint8).reshape(-1)).realize()
        layer = Linear(in_f, out_f, bias=False)
        layer.weight = ggml_data_to_tensor(raw, out_f * in_f, ggml_type).reshape(out_f, in_f).cast("float16")
        sample = np.unique(np.concatenate([np.arange(min(out_f, 32)), rng.integers(0, out_f, min(out_f, SAMPLE_ROWS))]))
        w = dequantize(blocks, args.type, sample, in_f)
        for rows in (int(r) for r in args.rows.split(",")):
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
            path = "tinyllm" if getattr(layer, "_tinyllm_q4", None) else "tinygrad"
            matrix_cores = rows % 16 == 0 and out_f % 16 == 0
            if path == "tinyllm" and matrix_cores:
                xr, wr = x_np.astype(np.float16).astype(np.float64), w.astype(np.float16).astype(np.float64)
            elif path == "tinyllm":
                xr, wr = quantize_activations(x_np), w
            else:
                xr, wr = x_np.astype(np.float64), w
            ref = xr @ wr.T
            err = float(np.max(np.abs(got.numpy()[:, sample] - ref) / (np.abs(xr) @ np.abs(wr).T + 1e-30)))
            tol = 2e-5 if path == "tinyllm" else 2e-3
            ms = statistics.median(times) * 1e3
            gbs = out_f * in_f // 32 * block_bytes / (ms / 1e3) / 1e9
            bad = not err <= tol
            failed |= bad
            flag = "  FAIL" if bad else ""
            line = f"{name:16s} {f'{out_f}x{in_f}':>12s} {rows:4d} {path:>9s} {ms:8.3f} {gbs:6.0f} {err:11.2e}{flag}"
            print(f"{args.type}: {line}")
        del raw, layer
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
