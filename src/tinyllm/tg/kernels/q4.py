"""Kernels for linear layers stored as Q4_0 or Q4_1.

At the pinned commit tinygrad has packed-weight kernels for the K, IQ and Q8_0 formats but none for
the legacy 4-bit formats, so a plain Q4_0 GGUF runs 89% of its weights through generic code
generation (docs/01-research-findings.md, section 5.5).

Both formats store 32 weights per block: an fp16 scale d (and for Q4_1 an fp16 offset m), then 16
bytes holding weight j in the low nibble and weight j+16 in the high nibble of byte j. A weight is
d * (q - 8) for Q4_0 and d * q + m for Q4_1. That is the layout of IQ4_NL, which tinygrad handles at
full speed; only the mapping from a 4-bit code to a value differs. These kernels therefore reuse
tinygrad's machinery with their own dequantizer:

  decode (up to 8 rows, or any count that is not a multiple of 16): activations quantized to int8 in
    groups of 32, 4-way integer dot products on the packed weights (tinygrad's `_decode_linear`);
  prefill (multiples of 16 rows): weights expanded to fp16 in registers and multiplied on the
    matrix cores (tinygrad's `_quant_linear_wmma`).
"""

from __future__ import annotations

import functools
import math
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import _amd_load, _is_cdna, _is_rdna4
from tinygrad.llm.kernels.amd_gguf import (
    _amd_dp4a,
    _decode_linear,
    _half,
    _load_u32,
    _nibble_lut,
    _quant_linear_wmma,
    _wmma_layout,
    _wmma_words,
    q8_quantize,
)
from tinygrad.uop.ops import Ops

from tinyllm.tg.kernels.dense import _unwrapped

GGML_Q4_0, GGML_Q4_1 = 2, 3
BLOCK_BYTES = {GGML_Q4_0: 18, GGML_Q4_1: 20}  # per 32 weights
QS_OFFSET = {GGML_Q4_0: 2, GGML_Q4_1: 4}  # byte offset of the nibbles in a block
NAMES = {GGML_Q4_0: "q4_0", GGML_Q4_1: "q4_1"}
# Q4_0 code n means n - 8, as int8 bytes: a 16-entry table for tinygrad's byte-permute lookup
Q4_0_LUT = (0xFBFAF9F8, 0xFFFEFDFC, 0x03020100, 0x07060504)
CHUNK = 1024  # inputs per wave in the decode kernel


def find_q4(decoded: Tensor, in_features: int) -> tuple[int, Tensor] | None:
    """(ggml type, blocks as uint16 halfwords) if `decoded` is exactly a Q4_0 or Q4_1 GGUF tensor, possibly cast.

    The halfwords are a zero-copy view of the model file's buffer (blocks are 2-byte aligned, not 4).
    """
    n = decoded.numel()
    if in_features % 32 or decoded.uop.axis is not None or not isinstance(n, int):
        return None
    graph = decoded.uop.toposort()
    target = _unwrapped(decoded.uop).key
    for ggml_type, block_bytes in BLOCK_BYTES.items():
        size = n // 32 * block_bytes
        raw = next(
            (
                u
                for u in graph
                if u.op in (Ops.SHRINK, Ops.BUFFER) and u.dtype == dtypes.uint8 and prod(u.shape) == size
            ),
            None,
        )
        if raw is None:
            continue
        if _unwrapped(ggml_data_to_tensor(Tensor(raw).flatten(), n, ggml_type).uop).key != target:
            continue
        offset = raw.contiguous_view_offset()
        if offset is None or offset % 2 or raw.buf_uop.dtype != dtypes.uint8:
            return None
        return ggml_type, Tensor(raw).flatten().bitcast(dtypes.uint16).contiguous()
    return None


def _codes(raw: UOp, base: UOp, i: int, ggml_type: int, stream: bool = False) -> UOp:
    """Weights 4i..4i+3 of the block at halfword `base`, one int8 per byte: q - 8 for Q4_0, q for Q4_1."""
    packed = _load_u32(raw, base, QS_OFFSET[ggml_type] + (i % 4) * 4, stream=stream)
    shift = 4 * (i // 4)  # weights 0..15 are low nibbles, 16..31 high nibbles
    if ggml_type == GGML_Q4_0:
        return _nibble_lut(packed, shift, Q4_0_LUT)
    return (packed >> shift) & 0x0F0F0F0F


@functools.cache
def _q4_decode_kernel(
    out: UOp, raw: UOp, xq: UOp, xd: UOp, xs: UOp, *, out_features: int, in_features: int, ggml_type: int, cdna: bool
) -> UOp:
    block_halfwords = BLOCK_BYTES[ggml_type] // 2
    token_tile = tokens if (tokens := int(out.shape[0])) <= 8 else 1

    def group_dot(token: UOp, output: UOp, group: UOp) -> UOp:
        # one block of 32 weights against one group of 32 int8 activations
        base = (output * (in_features // 32) + group) * block_halfwords
        xwords = _amd_load(xq[token, group, 0], 8)  # the group's 32 int8 activations, 4 per word
        dots = [UOp.const(0, dtypes.int32)] * 2
        for i in range(8):
            dots[i // 4] = _amd_dp4a(_codes(raw, base, i, ggml_type, stream=True), xwords[i], dots[i // 4], cdna)
        total = (dots[0] + dots[1]).float() * _half(raw[base])
        if ggml_type == GGML_Q4_1:
            # the offset m multiplies the sum of the group's activations, which q8_quantize keeps as two halves
            total = total + (xs[token, group, 0].load() + xs[token, group, 1].load()) * _half(raw[base + 1])
        return total * xd[token, group]

    return _decode_linear(out, out_features, in_features // 32, group_dot, f"linear_{NAMES[ggml_type]}", token_tile)


@functools.cache
def _q4_wmma_kernel(
    out: UOp, raw: UOp, x: UOp, *, out_features: int, in_features: int, ggml_type: int, kgroups: int, lanes: int
) -> UOp:
    tokens = int(out.shape[0])
    token_tile = 64 if tokens % 64 == 0 else 32 if tokens % 32 == 0 else 16
    layout = _wmma_layout(out, out_features, token_tile, 2, lanes)
    word_indices = _wmma_words(layout[5], kgroups)
    block_halfwords = BLOCK_BYTES[ggml_type] // 2

    def dequant(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...]:
        # `base` addresses a run of 8 blocks (256 weights); `subgroup` picks the block, `half` its 16 weights
        base = base + subgroup * block_halfwords
        scale = _half(raw[base])
        offset = _half(raw[base + 1]) if ggml_type == GGML_Q4_1 else UOp.const(0, dtypes.float32)
        words = tuple(_codes(raw, base, half * 4 + i, ggml_type) for i in word_indices)
        return tuple(
            ((word >> (b * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() * scale + offset).cast(dtypes.half)
            for word in words
            for b in range(4)
        )

    type_words = 8 * block_halfwords  # halfwords per 256 weights
    name = f"linear_{NAMES[ggml_type]}_f16_wmma"
    return _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, kgroups, lanes)


def q4_linear(layer: Any, x: Tensor) -> Tensor:
    """`layer(x)` for a Linear whose weights `find_q4` recognised, with a static token count."""
    ggml_type, raw = layer._tinyllm_q4
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    tokens = int(prod(x.shape[:-1]))
    x2 = x.reshape(tokens, in_features)
    cdna = _is_cdna(x.device)
    if tokens % 16 == 0 and out_features % 16 == 0:
        kgroups, lanes = (4, 64) if cdna else (2 if _is_rdna4(x.device) else 1, 32)
        fxn = functools.partial(
            _q4_wmma_kernel,
            out_features=out_features,
            in_features=in_features,
            ggml_type=ggml_type,
            kgroups=kgroups,
            lanes=lanes,
        )
        out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
        result = Tensor.custom_kernel(out, raw, x2.cast(dtypes.float16).contiguous(), fxn=fxn)[0]
    else:
        xq, xd, xs = q8_quantize(x2.contiguous(), tokens, in_features)
        fxn = functools.partial(
            _q4_decode_kernel, out_features=out_features, in_features=in_features, ggml_type=ggml_type, cdna=cdna
        )
        out = Tensor.empty(tokens, out_features, math.ceil(in_features / CHUNK), dtype=dtypes.float32, device=x.device)
        result = Tensor.custom_kernel(out, raw, xq, xd, xs, fxn=fxn)[0].sum(-1)
    result = result.reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
