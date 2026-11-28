#!/usr/bin/env python3
"""Correctness and speed of tinyllm's prefill kernels (Q4_K, Q5_K, IQ4_XS, Q6_K, Q8_0, BF16, F16) on Qwen3.8-27B
shapes.

    uv run python tools/prefill_kernel_check.py --type q8_0                     # tinyllm's kernel, all shapes
    uv run python tools/prefill_kernel_check.py --type q8_0 --variant tinygrad  # tinygrad's own path
    uv run python tools/prefill_kernel_check.py --type q6_k --quick             # small shapes
    uv run python tools/prefill_kernel_check.py --groups 8 --packed             # force a configuration
    uv run python tools/prefill_kernel_check.py --configs tinygrad,default,8p,2u --rows 32,128  # compare
    uv run python tools/prefill_kernel_check.py --type bf16 --configs 2b,2h,4b   # bf16 or fp16 instruction
    uv run python tools/prefill_kernel_check.py --no-weights [--no-inputs]      # diagnostics, errors expected
    uv run python tools/prefill_kernel_check.py --type q5_k --set "q45k.SHARED=(64,2,2)"  # experimental loops
    uv run python tools/prefill_kernel_check.py --rows 2048 --shapes "hc down=320x10240,delta out=2560x6144"

Blocks are random and wrapped exactly as tinygrad's GGUF loader wraps a tensor read from a file. The NumPy
reference reproduces each path's arithmetic: matrix-core rows (multiples of 16) use fp16 activations and the
path's fp16 rounding of the weights; other row counts go to tinygrad's decode kernel, which quantizes
activations to int8 per 32. BF16 and F16 weights are random 16-bit values as in tools/dense_kernel_check.py;
their decode rows go to tinyllm's dense kernel (fp32 activations, exact weights), their matrix-core rows
round the activations to bf16 or fp16 by mode, and tinygrad's own path for them is checked only loosely.
Errors are relative to the size of each dot product. Exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

SHAPES = [
    ("ffn gate/up", 17408, 5120),
    ("ffn down", 5120, 17408),
    ("attn q", 12288, 5120),
    ("deltanet qkv", 10240, 5120),
    ("deltanet out", 5120, 6144),
]
QUICK = [("small", 256, 1024), ("one block", 16, 256)]
SAMPLE_ROWS = 2048
Q6_K_BLOCK = np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)), ("sc", "i1", (16,)), ("d", "<f2")])
Q8_0_BLOCK = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
Q4_K_BLOCK = np.dtype([("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qs", "u1", (128,))])
Q5_K_BLOCK = np.dtype(
    [("d", "<f2"), ("dmin", "<f2"), ("scales", "u1", (12,)), ("qh", "u1", (32,)), ("qs", "u1", (128,))]
)
IQ4_XS_BLOCK = np.dtype([("d", "<f2"), ("scales_h", "<u2"), ("scales_l", "u1", (4,)), ("qs", "u1", (128,))])
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float64)
# How a path rounds a weight to fp16:
#   exact    no rounding (tinygrad's decode kernel multiplies the codes as integers)
#   tinygrad the exact value rounded once (fp32 arithmetic, then fp16)
#   tinyllm  the path's own fp16 arithmetic (per format, below)
Mode = str


def f16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float64)


def q6k_blocks(rng: np.random.Generator, n: int) -> np.ndarray:
    blocks = np.zeros(n // 256, dtype=Q6_K_BLOCK)
    blocks["ql"] = rng.integers(0, 256, (blocks.size, 128), dtype=np.uint8)
    blocks["qh"] = rng.integers(0, 256, (blocks.size, 64), dtype=np.uint8)
    blocks["sc"] = rng.integers(-128, 128, (blocks.size, 16), dtype=np.int8)
    blocks["d"] = rng.uniform(1e-5, 1e-4, blocks.size).astype(np.float16)
    return blocks


def q6k_weights(blocks: np.ndarray, rows: np.ndarray, inp: int, mode: Mode) -> np.ndarray:
    """(q - 32) * d * sc per weight, following ggml's dequantize_row_q6_K; tinyllm rounds d * sc to fp16 first."""
    b = blocks.reshape(-1, inp // 256)[rows]
    q = np.zeros((*b.shape, 256), dtype=np.int32)
    s = np.zeros((*b.shape, 256), dtype=np.float32)
    d = b["d"].astype(np.float32)[..., None]
    for n in range(2):
        ql, qh, sc = (
            b["ql"][..., 64 * n : 64 * n + 64],
            b["qh"][..., 32 * n : 32 * n + 32],
            b["sc"][..., 8 * n : 8 * n + 8],
        )
        lo, hi, qhi = ql[..., :32].astype(np.int32), ql[..., 32:].astype(np.int32), qh.astype(np.int32)
        for k, part in enumerate([lo & 15, hi & 15, lo >> 4, hi >> 4]):
            q[..., 128 * n + 32 * k : 128 * n + 32 * k + 32] = part | (((qhi >> (2 * k)) & 3) << 4)
            for half in range(2):
                at = 128 * n + 32 * k + 16 * half
                s[..., at : at + 16] = d * sc[..., 2 * k + half].astype(np.float32)[..., None]
    q, s = (q - 32).reshape(len(rows), inp).astype(np.float64), s.reshape(len(rows), inp).astype(np.float64)
    return q * s if mode == "exact" else f16(q * s) if mode == "tinygrad" else f16(q * f16(s))


def q8_blocks(rng: np.random.Generator, n: int) -> np.ndarray:
    blocks = np.zeros(n // 32, dtype=Q8_0_BLOCK)
    blocks["d"] = rng.uniform(1e-4, 1e-3, blocks.size).astype(np.float16)
    blocks["qs"] = rng.integers(-128, 128, (blocks.size, 32), dtype=np.int8)
    return blocks


def q8_weights(blocks: np.ndarray, rows: np.ndarray, inp: int, mode: Mode) -> np.ndarray:
    b = blocks.reshape(-1, inp // 32)[rows]
    q = b["qs"].astype(np.float64).reshape(len(rows), inp)
    s = np.repeat(b["d"].astype(np.float64), 32, axis=-1).reshape(len(rows), inp)
    return q * s if mode == "exact" else f16(q * s)  # d is fp16 already: one rounding on every path


def qk_blocks(dtype: np.dtype) -> Callable[[np.random.Generator, int], np.ndarray]:
    def make(rng: np.random.Generator, n: int) -> np.ndarray:
        blocks = np.zeros(n // 256, dtype=dtype)
        for field in dtype.names or ():
            if field in ("d", "dmin"):
                blocks[field] = rng.uniform(1e-5, 1e-4, blocks.size).astype(np.float16)
            else:
                blocks[field] = rng.integers(0, 256, blocks[field].shape, dtype=np.uint8)
        return blocks

    return make


def qk_weights(blocks: np.ndarray, rows: np.ndarray, inp: int, mode: Mode) -> np.ndarray:
    """d * sc * q - dmin * m per weight (ggml's dequantize_row_q4_K / q5_K); tinyllm rounds d * sc and dmin * m to
    fp16 once per group and the result once (fma)."""
    b = blocks.reshape(-1, inp // 256)[rows]
    s = b["scales"].astype(np.int64)
    sc = np.concatenate([s[..., 0:4] & 63, (s[..., 8:12] & 15) | ((s[..., 0:4] >> 6) << 4)], axis=-1)
    mn = np.concatenate([s[..., 4:8] & 63, (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)], axis=-1)
    qs = b["qs"].astype(np.int64).reshape(*b.shape, 4, 32)
    q = np.stack([qs & 15, qs >> 4], axis=-2).reshape(*b.shape, 8, 32)
    if "qh" in (b.dtype.names or ()):
        qh = b["qh"].astype(np.int64)[..., None, :]
        q = q + ((qh >> np.arange(8)[:, None]) & 1) * 16
    d, dmin = b["d"].astype(np.float64)[..., None, None], b["dmin"].astype(np.float64)[..., None, None]
    ds, dm = d * sc[..., None], dmin * mn[..., None]
    if mode == "tinyllm":
        w = f16(q * f16(ds) - f16(dm))
    else:
        w = q * ds - dm
        w = w if mode == "exact" else f16(w)
    return w.reshape(len(rows), inp)


def iq4xs_blocks(rng: np.random.Generator, n: int) -> np.ndarray:
    blocks = np.zeros(n // 256, dtype=IQ4_XS_BLOCK)
    blocks["d"] = rng.uniform(1e-5, 1e-4, blocks.size).astype(np.float16)
    blocks["scales_h"] = rng.integers(0, 1 << 16, blocks.size, dtype=np.uint16)
    blocks["scales_l"] = rng.integers(0, 256, (blocks.size, 4), dtype=np.uint8)
    blocks["qs"] = rng.integers(0, 256, (blocks.size, 128), dtype=np.uint8)
    return blocks


def iq4xs_weights(blocks: np.ndarray, rows: np.ndarray, inp: int, mode: Mode) -> np.ndarray:
    """d * (sc - 32) * kvalues_iq4nl[n] (ggml's dequantize_row_iq4_xs); tinyllm rounds d * (sc - 32) to fp16."""
    b = blocks.reshape(-1, inp // 256)[rows]
    g = np.arange(8)
    low = (b["scales_l"].astype(np.int64)[..., g // 2] >> (4 * (g % 2))) & 15
    high = (b["scales_h"].astype(np.int64)[..., None] >> (2 * g)) & 3
    ds = b["d"].astype(np.float64)[..., None] * ((low | (high << 4)) - 32)  # (..., 8)
    qs = b["qs"].astype(np.int64).reshape(*b.shape, 8, 16)
    v = KVALUES_IQ4NL[np.concatenate([qs & 15, qs >> 4], axis=-1)]  # (..., 8, 32)
    w = v * (f16(ds) if mode == "tinyllm" else ds)[..., None]
    return (w if mode == "exact" else f16(w)).reshape(len(rows), inp)


def dense_blocks(half: bool) -> Callable[[np.random.Generator, int], np.ndarray]:
    def make(rng: np.random.Generator, n: int) -> np.ndarray:
        """uint16 bit patterns of finite values between about 1e-5 and 0.5 in magnitude (some below fp16's normal
        range, 2^-14, for BF16)."""
        sign = rng.integers(0, 2, n, dtype=np.uint16) << 15
        if half:
            return sign | (rng.integers(0, 15, n, dtype=np.uint16) << 10) | rng.integers(0, 1024, n, dtype=np.uint16)
        return sign | (rng.integers(110, 126, n, dtype=np.uint16) << 7) | rng.integers(0, 128, n, dtype=np.uint16)

    return make


def f32_blocks(rng: np.random.Generator, n: int) -> np.ndarray:
    """uint32 bit patterns of fp32 values between about 1e-5 and 0.5 in magnitude."""
    sign = rng.integers(0, 2, n, dtype=np.uint32) << 31
    return sign | (rng.integers(110, 126, n, dtype=np.uint32) << 23) | rng.integers(0, 1 << 23, n, dtype=np.uint32)


def f32_weights(blocks: np.ndarray, rows: np.ndarray, inp: int, mode: Mode) -> np.ndarray:
    w = blocks.reshape(-1, inp)[rows].view(np.float32).astype(np.float64)
    return w if mode == "exact" else f16(w)  # every path rounds the weights to fp16, to nearest


def f16_toward_zero(a: np.ndarray) -> np.ndarray:
    """fp16 rounding toward zero (v_cvt_pkrtz_f16_f32), as float64."""
    near = a.astype(np.float16)
    over = np.abs(near.astype(np.float64)) > np.abs(a)
    return np.where(over, np.nextafter(near, np.float16(0)), near).astype(np.float64)


def dense_weights(half: bool) -> Callable[[np.ndarray, np.ndarray, int, Mode], np.ndarray]:
    def weights(blocks: np.ndarray, rows: np.ndarray, inp: int, mode: Mode) -> np.ndarray:
        bits = blocks.reshape(-1, inp)[rows]
        w = bits.view(np.float16).astype(np.float64) if half else (bits.astype(np.uint32) << 16).view(np.float32)
        w = w.astype(np.float64)
        return w if mode == "exact" or half else f16_toward_zero(w) if mode == "rtz" else f16(w)

    return weights


def bf16(a: np.ndarray) -> np.ndarray:
    """Round to bf16, nearest even, as float64."""
    bits = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return bits.astype(np.uint32).view(np.float32).astype(np.float64)


def _packed_tinyllm(packed: bool) -> Mode:
    return "tinyllm" if packed else "tinygrad"  # the unpacked variant keeps tinygrad's arithmetic


@dataclass(frozen=True)
class Format:
    ggml_type: int
    module: str
    kernel: str
    bytes_per_weight: float
    blocks: Callable[[np.random.Generator, int], np.ndarray]
    weights: Callable[[np.ndarray, np.ndarray, int, Mode], np.ndarray]
    tinyllm_mode: Callable[[bool], Mode]  # how tinyllm's kernel rounds, given whether its conversion is packed
    dense: bool = False  # BF16/F16: decode rows on tinyllm's dense kernel, matrix-core activations by mode


def dense_mode(ggml_type: int) -> Callable[[Any], Mode]:
    # densemm's mode: "bf16" keeps the weights exact and rounds activations to bf16, "f16" converts BF16 weights
    # toward zero to fp16; F16 weights are exact either way
    return lambda mode: "exact" if ggml_type == 1 or mode == "bf16" else "rtz"


FORMATS = {
    "q4_k": Format(12, "q45k", "q45k", 144 / 256, qk_blocks(Q4_K_BLOCK), qk_weights, _packed_tinyllm),
    "q5_k": Format(13, "q45k", "q45k", 176 / 256, qk_blocks(Q5_K_BLOCK), qk_weights, _packed_tinyllm),
    "iq4_xs": Format(23, "iq4xs", "iq4xs", 136 / 256, iq4xs_blocks, iq4xs_weights, _packed_tinyllm),
    "q6_k": Format(14, "q6k", "q6k", 210 / 256, q6k_blocks, q6k_weights, lambda packed: "tinyllm"),
    "q8_0": Format(8, "q8", "q8", 34 / 32, q8_blocks, q8_weights, lambda packed: "tinygrad"),
    "bf16": Format(30, "densemm", "dense", 2, dense_blocks(False), dense_weights(False), dense_mode(30), True),
    "f16": Format(1, "densemm", "dense", 2, dense_blocks(True), dense_weights(True), dense_mode(1), True),
    "f32": Format(0, "densemm", "f32", 4, f32_blocks, f32_weights, lambda mode: "tinygrad", True),
}


def quantize_activations(x: np.ndarray) -> np.ndarray:
    g = x.astype(np.float32).reshape(x.shape[0], -1, 32)
    scale = np.maximum(np.abs(g).max(-1, keepdims=True) / np.float32(127), np.float32(1e-8))
    return (np.clip(np.rint(g / scale), -127, 127) * scale).reshape(x.shape).astype(np.float64)


def module_mode(module: Any, rows: int, in_features: int, fmt: Format) -> Any:
    """The second item of the kernel module's configuration for this shape: packed conversion, or densemm's mode."""
    args = (rows, in_features, fmt.ggml_type) if fmt.module in ("q45k", "densemm") else (rows, in_features)
    return module._config(*args)[1]


CONVERSIONS = {"p": True, "u": False, "b": "bf16", "h": "f16"}


def parse_configs(text: str) -> list[tuple[str, int | None, Any, tuple[int, int] | None]]:
    """ "tinygrad", "default", or <groups><p|u|b|h>[@<token tile>x<output tiles>] (packed or unpacked conversion;
    for BF16 the bf16 or fp16 instruction; tiles where the kernel module accepts them), comma-separated."""
    configs: list[tuple[str, int | None, Any, tuple[int, int] | None]] = []
    for item in (c.strip() for c in text.split(",") if c.strip()):
        if item in ("tinygrad", "default"):
            configs.append((item, None, None, None))
        elif m := re.fullmatch(r"(\d+)([puhb])(?:@(\d+)x(\d+))?", item):
            tiles = (int(m[3]), int(m[4])) if m[3] else None
            configs.append((item, int(m[1]), CONVERSIONS[m[2]], tiles))
        else:
            raise SystemExit(f"unknown configuration {item!r}: use tinygrad, default or e.g. 8p, 2u, 2b, 2p@64x4")
    return configs


def set_conversion(module: Any, value: Any) -> None:
    """Apply a configuration's conversion letter to the kernel module (PACKED, or MODE for densemm)."""
    if hasattr(module, "MODE"):
        if isinstance(value, bool):
            raise SystemExit("BF16/F16 take b (bf16 instruction) or h (fp16 conversion), not p or u")
        module.MODE = value
    else:
        if isinstance(value, str):
            raise SystemExit("b and h apply to BF16/F16 only")
        module.PACKED = value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=sorted(FORMATS), default="q8_0")
    ap.add_argument("--variant", choices=("tinyllm", "tinygrad"), default="tinyllm")
    ap.add_argument("--configs", default=None, help="compare in one process, interleaved: e.g. tinygrad,default,8p,2u")
    ap.add_argument("--no-weights", action="store_true", help="diagnostic: constant weights; errors expected")
    ap.add_argument("--no-inputs", action="store_true", help="diagnostic: constant activations too (tinyllm's loop)")
    ap.add_argument("--groups", type=int, default=None, help="32-weight groups per loop step (default: by row count)")
    ap.add_argument("--packed", action=argparse.BooleanOptionalAction, default=None, help="packed fp16 conversion")
    ap.add_argument("--mode", choices=("bf16", "f16"), default=None, help="BF16: matrix instruction to use")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="only layers whose name contains this text")
    ap.add_argument("--shapes", default=None, help="layers in place of the 27B's: NAME=OUTxIN, comma-separated")
    ap.add_argument("--rows", default="1,8,32,128")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5, help="with --configs: timing rounds over all configurations")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="MODULE.NAME=VALUE",
        help="set a kernel module value, e.g. q45k.SHARED=(64,2,2) or wmma.NO_INPUTS=True (repeatable)",
    )
    args = ap.parse_args()
    fmt = FORMATS[args.type]
    if args.configs:
        configs = parse_configs(args.configs)
    elif args.variant == "tinygrad":
        configs = [("tinygrad", None, None, None)]
    else:
        configs = [("tinyllm", args.groups, args.mode if fmt.dense else args.packed, None)]

    from tinyllm import device

    device.configure("amd")
    import importlib

    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.kernels.amd import Linear

    from tinyllm import tg
    from tinyllm.tg import kernels

    tg.set_polling_wait()
    import ast

    for setting in args.set:  # applied before any kernel is built, as tools/profile_step.py --set does
        target, value = setting.split("=", 1)
        owner, attribute = target.rsplit(".", 1)
        setattr(importlib.import_module(f"tinyllm.tg.kernels.{owner}"), attribute, ast.literal_eval(value))
    module = importlib.import_module(f"tinyllm.tg.kernels.{fmt.module}")
    module.NO_WEIGHTS = args.no_weights
    from tinyllm.tg.kernels import wmma

    wmma.NO_INPUTS = args.no_inputs
    hooked = any(label != "tinygrad" for label, *_ in configs)
    if hooked:
        kernels.install()
    rounds = args.rounds if len(configs) > 1 else 1

    rng = np.random.default_rng(0)
    failed = False
    head = f"{'layer':16s} {'out x in':>12s} {'rows':>4s} {'path':>9s} {'ms':>8s} {'GB/s':>6s} {'TFLOPS':>6s}"
    print(f"{args.type} {'config':>8s}: {head} {'max rel err':>11s}")
    shapes = QUICK if args.quick else SHAPES
    if args.shapes:
        named = [item.split("=") for item in args.shapes.split(",")]
        shapes = [(name, *(int(n) for n in size.split("x"))) for name, size in named]
    for name, out_f, in_f in shapes:
        if args.only and args.only not in name:
            continue
        blocks = fmt.blocks(rng, out_f * in_f)
        raw = tg.Tensor(blocks.view(np.uint8).reshape(-1)).realize()
        layer = Linear(in_f, out_f, bias=False)
        layer.weight = ggml_data_to_tensor(raw, out_f * in_f, fmt.ggml_type).reshape(out_f, in_f).cast("float16")
        sample = np.unique(np.concatenate([np.arange(min(out_f, 32)), rng.integers(0, out_f, min(out_f, SAMPLE_ROWS))]))
        references: dict[Mode, np.ndarray] = {"exact": fmt.weights(blocks, sample, in_f, "exact")}
        for rows in (int(r) for r in args.rows.split(",")):
            x_np = rng.standard_normal((rows, in_f)).astype(np.float32)
            x = tg.Tensor(x_np).realize()
            layer(x)  # the hook's choice for this layer, before any configuration is captured
            matrix_cores = rows % 16 == 0 and out_f % 16 == 0
            if fmt.dense:  # small layers stay on the dense decode kernel
                matrix_cores = module.serves(layer, x)
            runs = []
            for label, groups, packed, tiles in configs:
                if label == "tinygrad":
                    call = Linear.__call__.__wrapped__ if hooked else Linear.__call__  # type: ignore[attr-defined]
                    jit = tg.TinyJit(lambda t, layer=layer, call=call: call(layer, t).realize())
                else:
                    module.GROUPS_PER_STEP = groups
                    set_conversion(module, packed)
                    if hasattr(module, "TILES"):
                        module.TILES = tiles
                    elif tiles:
                        raise SystemExit(f"{fmt.module} takes no tile override")
                    jit = tg.TinyJit(lambda t, layer=layer: layer(t).realize())
                for _ in range(3):  # capture with this configuration
                    got = jit(x)
                ours = label != "tinygrad" and getattr(layer, "_tinyllm_kernel", None) == fmt.kernel
                serves = getattr(module, "SERVES_ROWS", None)  # above it the module hands default runs to tinygrad
                ours = ours and not (serves is not None and rows > serves and groups is None and packed is None)
                path = "tinyllm" if (matrix_cores or fmt.dense) and ours else "tinygrad"
                tolerance = 2e-5
                if fmt.dense and path == "tinygrad":  # generic code: fp16 weights, activations not modelled
                    xr, mode, tolerance = x_np.astype(np.float64), "tinygrad", 2e-3
                elif fmt.dense and not matrix_cores:  # tinyllm's dense decode kernel: fp32 activations, exact weights
                    xr, mode = x_np.astype(np.float64), "exact"
                elif fmt.dense:
                    conversion = module_mode(module, rows, in_f, fmt)
                    xr = bf16(x_np) if conversion == "bf16" else x_np.astype(np.float16).astype(np.float64)
                    mode = fmt.tinyllm_mode(conversion)
                elif not matrix_cores:  # tinygrad's decode kernel: int8 activations, exact weights
                    xr, mode = quantize_activations(x_np), "exact"
                else:
                    xr = x_np.astype(np.float16).astype(np.float64)
                    mode = fmt.tinyllm_mode(module_mode(module, rows, in_f, fmt)) if path == "tinyllm" else "tinygrad"
                if mode not in references:
                    references[mode] = fmt.weights(blocks, sample, in_f, mode)
                wr = references[mode]
                err = float(np.max(np.abs(got.numpy()[:, sample] - xr @ wr.T) / (np.abs(xr) @ np.abs(wr).T + 1e-30)))
                runs.append((label, path, jit, err, tolerance, []))
            module.GROUPS_PER_STEP = None
            for setting in ("PACKED", "MODE"):
                if hasattr(module, setting):
                    setattr(module, setting, None)
            if hasattr(module, "TILES"):
                module.TILES = None
            for _ in range(rounds):
                for _label, _path, jit, _err, _tol, times in runs:
                    for _ in range(max(1, args.iters // rounds)):
                        start = time.perf_counter()
                        jit(x)
                        tg.synchronize()
                        times.append(time.perf_counter() - start)
            for label, path, _jit, err, tolerance, times in runs:
                ms = statistics.median(times) * 1e3
                gbs = out_f * in_f * fmt.bytes_per_weight / (ms / 1e3) / 1e9
                tflops = 2 * rows * out_f * in_f / (ms / 1e3) / 1e12
                bad = not err <= tolerance
                failed |= bad
                line = f"{name:16s} {f'{out_f}x{in_f}':>12s} {rows:4d} {path:>9s} {ms:8.3f} {gbs:6.0f} {tflops:6.1f}"
                print(f"{args.type} {label:>8s}: {line} {err:11.2e}{'  FAIL' if bad else ''}", flush=True)
        del raw, layer
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
