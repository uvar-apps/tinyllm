#!/usr/bin/env python3
"""Correctness and speed of tinyllm's Q5_0/Q5_1/Q2_0 expert kernels on the GPU, on Qwen3.8-Flash-Next shapes.

    uv run python tools/moe_kernel_check.py                    # Q5_1, the down projection, 1, 4 and 32 tokens
    uv run python tools/moe_kernel_check.py --type q5_0
    uv run python tools/moe_kernel_check.py --quick            # small shapes, every code path
    uv run python tools/moe_kernel_check.py --baseline         # tinygrad's own path, for comparison

Blocks are random (fp16 scales, random codes) and wrapped exactly as tinygrad's GGUF loader wraps an expert
tensor read from a file. Each token picks `--slots` experts at random. The activations come either one row per
token (as the gate and up projections take them) or one row per slot (the down projection). The kernels keep
the activations in fp32, the NumPy reference in float64; tinygrad's own path rounds the weights to fp16. Errors
are relative to the size of each dot product. Bandwidth counts the bytes of the experts read once per slot.
Exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

SHAPES = [("ffn down", 2560, 640, 512), ("gate/up shape", 640, 2560, 64)]  # name, out, in, experts
QUICK = [("small", 64, 128, 8), ("ragged chunk", 96, 1344, 4), ("odd rows", 100, 640, 16)]
TYPES = {"q5_0": (6, 32, 22), "q5_1": (7, 32, 24), "q2_0": (42, 64, 18)}  # ggml type, weights, bytes


def random_blocks(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    _, elems, _ = TYPES[kind]
    fields = [("d", "<f2")] + ([("m", "<f2")] if kind == "q5_1" else [])
    fields += [("qh", "<u4"), ("qs", "u1", (16,))] if kind != "q2_0" else [("qs", "u1", (16,))]
    blocks = np.zeros(n // elems, dtype=np.dtype(fields))
    blocks["d"] = rng.uniform(0.001, 0.01, blocks.size).astype(np.float16)
    if kind == "q5_1":
        blocks["m"] = rng.uniform(-0.05, 0.05, blocks.size).astype(np.float16)
    if kind != "q2_0":
        blocks["qh"] = rng.integers(0, 2**32, blocks.size, dtype=np.uint32)
    blocks["qs"] = rng.integers(0, 256, (blocks.size, 16), dtype=np.uint8)
    return blocks


def dequantize(blocks: np.ndarray, kind: str) -> np.ndarray:
    """(blocks,) -> (blocks, weights per block) float64, ggml's layout."""
    d = blocks["d"].astype(np.float64)[:, None]
    if kind == "q2_0":
        q = (blocks["qs"][:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
        return d * (q.reshape(-1, 64).astype(np.float64) - 1)
    low = np.concatenate([blocks["qs"] & 0x0F, blocks["qs"] >> 4], axis=-1).astype(np.int64)
    high = (blocks["qh"][:, None] >> np.arange(32, dtype=np.uint32)) & 1
    q = (low | (high.astype(np.int64) << 4)).astype(np.float64)
    return d * (q - 16) if kind == "q5_0" else d * q + blocks["m"].astype(np.float64)[:, None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=sorted(TYPES), default="q5_1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="do not install tinyllm's kernels")
    ap.add_argument("--tokens", default="1,4,32")
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from tinyllm import device

    device.configure("amd")
    from tinygrad.llm import gguf as tg_gguf
    from tinygrad.llm.kernels.amd import ExpertWeights

    from tinyllm import tg
    from tinyllm.tg import kernels
    from tinyllm.tg.kernels import moe_small

    tg.set_polling_wait()
    moe_small.install_decoders()
    if not args.baseline:  # the one-row kernels at every token count: the grouped ones are moe_mm_check.py's
        kernels.install([k for k in kernels.KERNELS if k != "moe_mm"])

    ggml_type, elems, block_bytes = TYPES[args.type]
    rng = np.random.default_rng(0)
    slots = min(args.slots, 3) if args.quick else args.slots
    failed = False
    head = f"{'layer':14s} {'out x in':>10s} {'exp':>4s} {'tok':>4s} {'x':>4s} {'path':>9s} {'ms':>8s} {'GB/s':>6s}"
    head += f" {'max rel err':>11s}"
    print(f"{args.type}: {head}")
    for name, out_f, in_f, experts in QUICK if args.quick else SHAPES:
        blocks = random_blocks(rng, experts * out_f * in_f, args.type)
        raw = tg.Tensor(blocks.view(np.uint8).reshape(-1)).realize()
        layer = ExpertWeights(experts, in_f, out_f)
        shape = (experts, out_f, in_f)
        layer.weight = (
            tg_gguf.ggml_data_to_tensor(raw, experts * out_f * in_f, ggml_type).reshape(shape).cast("float16")
        )
        w = dequantize(blocks, args.type).reshape(experts, out_f, in_f)
        for tokens in (int(t) for t in args.tokens.split(",")):
            for x_slots in (1, slots):
                sel_np = np.stack([rng.choice(experts, slots, replace=False) for _ in range(tokens)]).astype(np.int32)
                x_np = rng.standard_normal((1, tokens, x_slots, in_f)).astype(np.float32)
                sel, x = tg.Tensor(sel_np.reshape(1, tokens, slots)).realize(), tg.Tensor(x_np).realize()
                jit = tg.TinyJit(lambda s, t, layer=layer: layer(s, t).realize())
                got = None
                for _ in range(3):
                    got = jit(sel, x)
                times = []
                for _ in range(args.iters):
                    start = time.perf_counter()
                    got = jit(sel, x)
                    tg.synchronize()
                    times.append(time.perf_counter() - start)
                path = "tinyllm" if layer.__dict__.get("_tinyllm_moe") else "tinygrad"
                xs = x_np.reshape(tokens, x_slots, in_f)
                result = got.numpy().reshape(tokens, slots, out_f)
                err = 0.0
                for t in range(tokens):
                    for s in range(slots):
                        row = xs[t, s if x_slots > 1 else 0][None]
                        xr = row.astype(np.float64)
                        we = w[sel_np[t, s]]
                        if path == "tinyllm":
                            ref, size = xr @ we.T, np.abs(xr) @ np.abs(we).T
                        else:  # tinygrad's path rounds the weights to fp16
                            wh = we.astype(np.float16).astype(np.float64)
                            ref, size = xr @ wh.T, np.abs(xr) @ np.abs(wh).T
                        err = np.maximum(err, float(np.max(np.abs(result[t, s] - ref[0]) / (size[0] + 1e-30))))
                tol = 2e-5 if path == "tinyllm" else 2e-3
                ms = statistics.median(times) * 1e3
                gbs = tokens * slots * out_f * in_f // elems * block_bytes / (ms / 1e3) / 1e9
                bad = not err <= tol
                failed |= bad
                flag = "  FAIL" if bad else ""
                line = (
                    f"{name:14s} {f'{out_f}x{in_f}':>10s} {experts:4d} {tokens:4d} {x_slots:4d} {path:>9s} "
                    f"{ms:8.3f} {gbs:6.0f} {err:11.2e}{flag}"
                )
                print(f"{args.type}: {line}")
        del raw, layer
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
