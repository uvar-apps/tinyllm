#!/usr/bin/env python3
"""Grouped expert prefill (tinyllm.tg.kernels.moe_mm) against float64, and its speed against the one-row kernels.

    DEV=KFD+AMD uv run python tools/moe_mm_check.py [--types q4_k,q5_1] [--tokens 33,128,2048] [--quick]
    DEV=KFD+AMD uv run python tools/moe_mm_check.py --no-grouped     # the one-row kernels, for comparison

Blocks are random bytes with small fp16 scales, wrapped exactly as tinygrad's GGUF loader wraps an expert tensor
read from a file; the reference weights are tinygrad's own decoding of them. Each token picks `--slots` experts at
random. Gate/up layers take one activation row per token, down layers one per slot, at Qwen3.8-Flash-Next's
shapes (512 experts; 2,560 -> 640 and 640 -> 2,560). The grouped kernel rounds activations and weights to fp16, so
errors are relative to the size of each dot product (|x| . |w|). Exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

# ggml type, weights and bytes per block, byte offsets of fp16 scale fields (value range)
TYPES = {
    "q4_k": (12, 256, 144, ((0, 0.001, 0.01), (2, 0.001, 0.01))),
    "q5_k": (13, 256, 176, ((0, 0.001, 0.01), (2, 0.001, 0.01))),
    "q8_0": (8, 32, 34, ((0, 0.001, 0.01),)),
    "q5_1": (7, 32, 24, ((0, 0.001, 0.01), (2, -0.05, 0.05))),
    "iq4_xs": (23, 256, 136, ((0, 0.001, 0.01),)),
    "iq3_xxs": (18, 256, 98, ((0, 0.001, 0.01),)),
    "iq2_s": (22, 256, 82, ((0, 0.001, 0.01),)),
    "iq4_nl": (20, 32, 18, ((0, 0.001, 0.01),)),
    "q5_0": (6, 32, 22, ((0, 0.001, 0.01),)),
    "q2_0": (42, 64, 18, ((0, 0.001, 0.01),)),
}
LAYERS = [("gate/up", 640, 2560, 1), ("down", 2560, 640, 0)]  # name, out, in, one activation row per token
QUICK = [("gate/up", 128, 512, 1), ("down", 256, 256, 0)]


def random_raw(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    _, elems, nbytes, fields = TYPES[kind]
    blocks = rng.integers(0, 256, (n // elems, nbytes), dtype=np.uint8)
    for offset, lo, hi in fields:
        blocks[:, offset : offset + 2] = (
            rng.uniform(lo, hi, len(blocks)).astype(np.float16).view(np.uint8).reshape(-1, 2)
        )
    return blocks.reshape(-1)


def selections(
    rng: np.random.Generator, tokens: int, slots: int, experts: int, counts: np.ndarray | None = None
) -> np.ndarray:
    """(tokens, slots) distinct experts per token: uniform at random, or with `counts` (rows per expert of a real
    layer) scaled to tokens * slots pairs, every expert at most once per token."""
    if counts is None:
        return np.stack([rng.choice(experts, slots, replace=False) for _ in range(tokens)])
    want = counts.astype(np.float64) * (tokens * slots) / counts.sum()
    n = np.minimum(np.floor(want).astype(np.int64), tokens)
    while n.sum() < tokens * slots:  # largest remainders, never more rows than tokens
        n[np.argmax(np.where(n < tokens, want - n, -np.inf))] += 1
    ids = np.repeat(np.arange(len(n)), n)  # sorted: an expert's pairs are fewer than `tokens` apart
    return ids.reshape(slots, tokens).T[rng.permutation(tokens)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument(
        "--routing",
        metavar="NPY",
        help="rows per expert of real layers, (calls, layers, experts) as a routing dump saves them: each check "
        "takes the counts of one layer (--layer), scaled to its pairs, instead of uniform random experts",
    )
    ap.add_argument("--layer", type=int, default=60, help="which layer of --routing (calls and layers flattened)")
    ap.add_argument(
        "--skewed",
        action="store_true",
        help="rows per expert falling as 1 / rank, as real routing does (a few experts take hundreds of rows, "
        "so they need several tiles) instead of uniform random experts",
    )
    ap.add_argument("--tokens", default=None, help="token counts (default 33,128,2048; with --quick 33,64)")
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="small shapes, few experts, every code path")
    ap.add_argument("--no-grouped", action="store_true", help="the one-row kernels instead, for comparison")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a moe_mm setting")
    ap.add_argument(
        "--fused", action="store_true", help="gate and up layers fused, SwiGLU out (moe_mm.grouped_gate_up)"
    )
    args = ap.parse_args()

    from tinyllm import device

    device.configure("amd")
    from tinygrad.llm import gguf as tg_gguf
    from tinygrad.llm.kernels.amd import ExpertWeights

    from tinyllm import tg
    from tinyllm.tg import kernels
    from tinyllm.tg.kernels import moe_mm, moe_small

    for setting in args.set:
        name, value = setting.split("=", 1)
        setattr(moe_mm, name, int(value) if value.lstrip("-").isdigit() else value)
    tg.set_polling_wait()
    moe_small.install_decoders()
    kernels.install([k for k in kernels.KERNELS if not (args.no_grouped and k == "moe_mm")])
    rng = np.random.default_rng(0)
    experts = 16 if args.quick else args.experts
    tokens_arg = args.tokens or ("33,64" if args.quick else "33,128,2048")
    token_counts = [int(t) for t in tokens_arg.split(",")]
    args.counts = None
    if args.skewed:  # the shares of a Zipf law over the experts, in a shuffled order
        args.counts = rng.permutation(1e6 / np.arange(1, experts + 1))
    if args.routing:
        counts = np.load(args.routing)
        counts = counts.reshape(-1, counts.shape[-1])[args.layer]
        if len(counts) != experts:
            raise SystemExit(f"--routing has {len(counts)} experts, the check {experts}")
        args.counts = counts
    failed = False
    print(
        f"{'type':5s} {'layer':8s} {'out x in':>10s} {'tok':>5s} {'path':>8s} {'ms':>8s} {'TFLOPS':>7s} {'max err':>9s}"
    )
    if args.fused:
        for kind in args.types.split(","):
            failed |= fused_check(args, kind, rng, experts, token_counts)
        print(f"tinyllm moe_mm check: {'FAIL' if failed else 'all ok'}")
        return 1 if failed else 0
    for kind in args.types.split(","):
        ggml_type, elems, _, _ = TYPES[kind]
        for name, out_f, in_f, shared in QUICK if args.quick else LAYERS:
            if in_f % elems:
                continue
            n = experts * out_f * in_f
            raw = tg.Tensor(random_raw(rng, n, kind)).realize()
            decoded = tg_gguf.ggml_data_to_tensor(raw, n, ggml_type).reshape(experts, out_f, in_f)
            w = decoded.float().numpy().astype(np.float64)
            for tokens in token_counts:
                layer = ExpertWeights(experts, in_f, out_f)
                layer.weight = decoded.cast("float16")
                x_slots = 1 if shared else args.slots
                sel_np = selections(rng, tokens, args.slots, experts, args.counts)
                x_np = rng.standard_normal((1, tokens, x_slots, in_f)).astype(np.float32)
                sel = tg.Tensor(sel_np.astype(np.int32).reshape(1, tokens, args.slots)).realize()
                x = tg.Tensor(x_np).realize()
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
                path = "grouped" if layer.__dict__.get("_tinyllm_moe_mm") and tokens >= moe_mm.MIN_TOKENS else "one-row"
                result = got.numpy().reshape(tokens, args.slots, out_f)
                xs = x_np.reshape(tokens, x_slots, in_f).astype(np.float64)
                err = 0.0
                for t in range(tokens if tokens <= 128 else 128):
                    for s in range(args.slots):
                        row, we = xs[t, s if x_slots > 1 else 0], w[sel_np[t, s]]
                        ref, size = we @ row, np.abs(we) @ np.abs(row)
                        err = np.maximum(err, float(np.max(np.abs(result[t, s] - ref) / (size + 1e-30))))
                ms = statistics.median(times) * 1e3
                tflops = 2 * tokens * args.slots * out_f * in_f / (ms / 1e3) / 1e12
                bad = not err <= 2e-3
                failed |= bad
                print(
                    f"{kind:5s} {name:8s} {f'{out_f}x{in_f}':>10s} {tokens:5d} {path:>8s} {ms:8.3f} {tflops:7.2f} "
                    f"{err:9.2e}{'  FAIL' if bad else ''}"
                )
            del raw, decoded
    print(f"tinyllm moe_mm check: {'FAIL' if failed else 'all ok'}")
    return 1 if failed else 0


def fused_check(args: argparse.Namespace, kind: str, rng: np.random.Generator, experts: int, counts: list[int]) -> bool:
    """grouped_gate_up against float64 SwiGLU(gate x, up x) on two random layers of `kind`."""
    from tinygrad.llm import gguf as tg_gguf
    from tinygrad.llm.kernels.amd import ExpertWeights

    from tinyllm import tg
    from tinyllm.tg.kernels import moe_mm

    ggml_type = TYPES[kind][0]
    out_f, in_f = (128, 512) if args.quick else (640, 2560)
    n = experts * out_f * in_f
    layers, weights = [], []
    for _ in range(2):
        raw = tg.Tensor(random_raw(rng, n, kind)).realize()
        decoded = tg_gguf.ggml_data_to_tensor(raw, n, ggml_type).reshape(experts, out_f, in_f)
        layer = ExpertWeights(experts, in_f, out_f)
        layer.weight = decoded.cast("float16")
        layer._tinyllm_moe_mm = moe_mm.find(layer)
        layers.append(layer)
        # the kernel's weights: decoded, then rounded to fp16
        weights.append(decoded.float().numpy().astype(np.float16).astype(np.float64))
    failed = False
    for tokens in counts:
        sel_np = selections(rng, tokens, args.slots, experts, args.counts)
        # small activations: random blocks give large weights, and SwiGLU(gate, up) must stay in fp16's range
        x_np = (0.05 * rng.standard_normal((1, tokens, 1, in_f))).astype(np.float32)
        sel = tg.Tensor(sel_np.astype(np.int32).reshape(1, tokens, args.slots)).realize()
        x = tg.Tensor(x_np).realize()
        jit = tg.TinyJit(lambda s, t: moe_mm.grouped_gate_up(layers[0], layers[1], s, t).realize())
        for _ in range(3):
            got = jit(sel, x)
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            got = jit(sel, x)
            tg.synchronize()
            times.append(time.perf_counter() - start)
        result = got.float().numpy().reshape(tokens, args.slots, out_f)
        err = 0.0
        for t in range(min(tokens, 128)):
            row = x_np[0, t, 0].astype(np.float16).astype(np.float64)  # the kernel's activations, in fp16
            for s in range(args.slots):
                g, u = weights[0][sel_np[t, s]] @ row, weights[1][sel_np[t, s]] @ row
                ref = g / (1 + np.exp(-g)) * u
                err = np.maximum(err, float(np.max(np.abs(result[t, s] - ref)) / (np.max(np.abs(ref)) + 1e-30)))
        ms = statistics.median(times) * 1e3
        tflops = 4 * tokens * args.slots * out_f * in_f / (ms / 1e3) / 1e12
        # fp32 sums in another order and the fp16 result. Types whose block scale is a product (d * sub-scale) round
        # it to fp16 and then code * scale again: each weight is rounded twice (measured up to 2.4e-3 for the IQ types)
        bad = not err <= (4e-3 if kind in ("q4_k", "q5_k", "iq4_xs", "iq3_xxs", "iq2_s") else 2e-3)
        failed |= bad
        print(
            f"{kind:5s} {'gate+up':8s} {f'{out_f}x{in_f}':>10s} {tokens:5d} {'fused':>8s} {ms:8.3f} {tflops:7.2f} "
            f"{err:9.2e}{'  FAIL' if bad else ''}"
        )
    return failed


if __name__ == "__main__":
    sys.exit(main())
