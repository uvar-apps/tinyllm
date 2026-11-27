"""Matrix-core prefill kernel for linear layers stored as IQ4_XS.

In a 32-token prefill of `Qwen3.8-27B-UD-Q4_K_XL.gguf`, tinygrad's matrix-core kernel for IQ4_XS takes 17% of the
pass (`tools/profile_step.py --kind prefill`), on tinygrad's one-group-per-step loop. tinygrad's decode kernel is
kept. This kernel runs tinygrad's own arithmetic on tinyllm's loop (wmma.py) with 2 groups of 32 weights per
step, so the weights the matrix cores see, and the outputs, are bit-identical to tinygrad's (`_config`).

IQ4_XS super-block, 256 weights in 136 bytes (34 words): word 0 holds the fp16 scale d and the high two bits of
the eight 6-bit group scales, word 1 their low four bits, then 16 bytes per group of 32. Byte j of a group holds
the code of weight j in its low nibble and of weight j + 16 in its high nibble. A code n means the int8 value
kvalues_iq4nl[n], and a weight is d * (sc - 32) * kvalues_iq4nl[n].

Decode (up to 8 rows): tinygrad's integer-dot kernel, with one change (`_iq4xs_decode_kernel`). tinygrad computes a
group's total as dot * (d * (sc - 32)); its C renderer prints a product inlined into a product without
parentheses, so with one row the kernel evaluates ((dot * d) * (sc - 32)), while with several rows d * (sc - 32)
is shared between rows and kept as written. The rows of a several-row call then differ from one-row calls by a few
units in the last place, which a speculative verify (tinyllm.tg.speculative) must not. Here the product is written
left to right, so every row count evaluates what tinygrad's one-row kernel does.

As in tinygrad, a 256-entry table in local memory maps a code byte to the two fp16 values of its nibbles. Both
halves of a group (weights 0-15, 16-31) are converted in the same loop step, so one table load serves two
weights. A packed conversion is kept as an option (PACKED = True): one v_perm_b32 collects the low (or high)
values of two bytes into one word and one v_pk_mul_f16 scales both, with d * (sc - 32) rounded to fp16 once per
group. It is about 2.5% faster for these layers but rounds a weight twice where tinygrad rounds once.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.llm.kernels.amd import _amd_load, _is_cdna, _is_rdna4
from tinygrad.llm.kernels.amd_gguf import (
    IQ4_XS,
    QUANT_SIZES,
    _amd_byte_perm,
    _amd_dp4a,
    _decode_linear,
    _half,
    _iq4_scale,
    _quant_word,
    _wmma_layout,
    _wmma_words,
    iq4_half_lut,
    q8_quantize,
)

from tinyllm.tg.kernels.wmma import bind_rows, scale_pair, splat_half, valid_rows, wmma_linear

QS_WORD = 2  # first word of the codes in a super-block
HALF_SELECTORS = (0x05040100, 0x07060302)  # v_perm_b32: the low (then high) fp16 values of two table entries
# Overrides for experiments (tools/prefill_kernel_check.py); None selects by row count, see `_config`.
GROUPS_PER_STEP: int | None = None
PACKED: bool | None = None
NO_WEIGHTS = False  # diagnostic: constant weights, no weight loads. Wrong results; shows what the rest costs.
LDS_BYTES: int | None = None  # experiments: local memory reserved per workgroup instead of RESERVE_LDS
TILES: tuple[int, int] | None = None  # experiments: (token tile, output tiles per wave) instead of the default
RESERVE_LDS = 0  # local memory reserved per workgroup (wmma.wmma_stores); none, see `_config`
SERVES_ROWS = 32  # above this many rows tinygrad's own kernel is as fast or faster (`_config`); None: all rows


def _config(tokens: int, in_features: int) -> tuple[int, bool]:
    """(groups per step, packed conversion). Measured inside the model, a 32-token prefill of `UD-Q4_K_XL` with
    fixed-length chunks, ms per pass in this kernel (tools/profile_step.py --set): tinygrad 27.9, 1 group packed
    27.4, 2 groups unpacked 22.9, 2 packed 22.4, 4 packed 26.7, 8 packed 30.8. Two groups with tinygrad's
    arithmetic keep the results exact for 0.6 ms more than packed.

    Above 32 rows tinygrad's tiling holds 128 tokens per wave for the wide layers, and more groups per step
    then lose even in isolation (feed-forward down at 128 rows: 1.19 ms with tinygrad's kernel, 1.86 with
    2 groups packed); `iq4xs_linear` leaves those row counts to tinygrad's kernel (`SERVES_ROWS`).

    Local memory reserved per workgroup to limit occupancy (`RESERVE_LDS`; densemm.py explains why that helps),
    ms per pass in this kernel inside the model: chunk 32, none 22.8, 16 KB 19.4, 24 KB 21.1, 32 KB 29.9 (with the
    same reservation for Q4_K and Q5_K). With 16 KB here and none for Q4_K and Q5_K, 21.4 and 21.7 against 22.9
    and 23.1 (twice each), but the whole pass moved by under 1 ms, so nothing is reserved. At chunk 128 every
    reservation stayed slower than tinygrad's kernel (97.6 ms or more against 84.0).
    """
    groups = GROUPS_PER_STEP or 2
    packed = False if PACKED is None else PACKED
    return (groups if (in_features // 32) % groups == 0 else 1), packed


def _tiles(tokens: int, out_features: int) -> tuple[int, int]:
    """(token tile, output tiles): tinygrad's choice for IQ4_XS (_iq4_linear_f16_wmma_kernel)."""
    if out_features <= 1024 and tokens % 32 == 0:
        token_tile = 32
    elif tokens % 64 == 0 and out_features <= 6144:
        token_tile = 64
    elif tokens % 128 == 0:
        token_tile = 128
    else:
        token_tile = 32 if tokens % 32 == 0 else 16
    output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
    return token_tile, output_tiles


@functools.cache
def _iq4xs_wmma_kernel(
    out: UOp,
    raw: UOp,
    x: UOp,
    table: UOp,
    *,
    out_features: int,
    in_features: int,
    kgroups: int,
    lanes: int,
    groups_per_step: int,
    packed: bool,
    no_weights: bool = False,
    lds_bytes: int = 0,
    tiles: tuple[int, int] | None = None,
    rows: Any = None,
) -> UOp:
    token_tile, output_tiles = _tiles(int(out.shape[0]), out_features)
    if tiles is not None and int(out.shape[0]) % tiles[0] == 0:
        token_tile, output_tiles = tiles
    layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes)
    output_waves, _, _, lane, wave, lane_group, _, _, _ = layout
    word_indices = _wmma_words(lane_group, kgroups)
    # the table into local memory, as tinygrad does
    local = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, items = wave * lanes + lane, 256 // (lanes * output_waves)
    lut = local.after(*(local[tid * items + i].store(table[tid * items + i]) for i in range(items)))

    def dequant(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
        if no_weights:
            return tuple(UOp.const(0.001 * (k + 1), dtypes.half) for k in range(4 * len(word_indices)))
        scale = _iq4_scale(raw, base, subgroup)
        words = [raw[base + QS_WORD + subgroup * 4 + i] for i in word_indices]
        # entry for a code byte: fp16 value of its low nibble in bits 0-15, of its high nibble in bits 16-31
        entries = [lut[((w >> (8 * b)) & 255).cast(dtypes.weakint)].load() for w in words for b in range(4)]
        if packed:
            scale2 = splat_half(scale.cast(dtypes.half))
            selector = UOp.const(HALF_SELECTORS[half], dtypes.uint32)
            pairs = [_amd_byte_perm(entries[j + 1], entries[j], selector) for j in range(0, len(entries), 2)]
            return UOp.stack(*(scale_pair(p, scale2) for p in pairs)).bitcast(dtypes.half)
        return tuple((_half((e >> (half * 16)) & 0xFFFF) * scale).cast(dtypes.half) for e in entries)

    name = (
        f"linear_iq4_xs_f16_wmma_g{groups_per_step}{'_packed' if packed else ''}{'_noweights' if no_weights else ''}"
        + (f"_lds{lds_bytes}" if lds_bytes else "")
        + (f"_t{tiles[0]}x{tiles[1]}" if tiles else "")
    )
    type_words = QUANT_SIZES[IQ4_XS] // 4
    return wmma_linear(
        out,
        x,
        out_features,
        in_features,
        type_words,
        layout,
        dequant,
        name,
        kgroups,
        lanes,
        groups_per_step,
        split_blocks=8 % groups_per_step == 0,
        lds_bytes=lds_bytes,
        rows=rows,
    )


def find_iq4xs(layer: Any) -> tuple[int, Tensor] | None:
    """tinygrad's own packed form of the layer's weights if they are IQ4_XS, else None."""
    from tinygrad.llm.kernels.amd_gguf import find_packed

    found = find_packed(layer.weight, int(layer.in_features))
    return found if found is not None and found[0] == IQ4_XS else None


DECODE_ROWS = 8  # up to this many rows (not a multiple of 16) take `_iq4xs_decode_kernel`


@functools.cache
def _iq4xs_decode_kernel(
    out: UOp, raw: UOp, xq: UOp, xd: UOp, xs: UOp, *, out_features: int, in_features: int, cdna: bool
) -> UOp:
    """tinygrad's `_quant_decode_kernel` for IQ4_XS with the group total written left to right (module notes)."""
    token_tile = tokens if (tokens := int(out.shape[0])) <= 8 else 1
    words = QUANT_SIZES[IQ4_XS] // raw.dtype.itemsize

    def group_dot(token: UOp, output: UOp, group: UOp) -> UOp:
        block, subgroup = group // 8, group % 8
        base = (output * in_features // 256 + block) * words
        xwords = _amd_load(xq[token, group, 0], 8)
        dot = UOp.const(0, dtypes.int32)
        for i in range(8):
            dot = _amd_dp4a(_quant_word(raw, base, subgroup, i, IQ4_XS, None), xwords[i], dot, cdna)
        header, low = raw[base].load(), raw[base + 1].load()
        scale = ((low >> (4 * subgroup)) & 15) | (((header >> (16 + 2 * subgroup)) & 3) << 4)
        return dot.float() * _half(header) * (scale.cast(dtypes.int32) - 32).float() * xd[token, group]

    return _decode_linear(out, out_features, in_features // 32, group_dot, "linear_iq4_xs", token_tile)


def _iq4xs_decode(layer: Any, x: Tensor, tokens: int, in_features: int, out_features: int) -> Tensor:
    """What tinygrad's `q8_linear` does for a few rows of an IQ4_XS layer, with `_iq4xs_decode_kernel`."""
    xq, xd, xs = q8_quantize(x.reshape(tokens, in_features).contiguous(), tokens, in_features)
    fxn = functools.partial(
        _iq4xs_decode_kernel, out_features=out_features, in_features=in_features, cdna=_is_cdna(x.device)
    )
    chunks = (in_features + 1023) // 1024
    out = Tensor.empty(tokens, out_features, chunks, dtype=dtypes.float32, device=x.device)
    result = Tensor.custom_kernel(out, layer.weight, xq, xd, xs, fxn=fxn)[0].sum(-1)
    result = result.reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias


def iq4xs_linear(layer: Any, x: Tensor, fallback: Callable[[Any, Tensor], Tensor]) -> Tensor:
    """`layer(x)` for an IQ4_XS layer: matrix cores for multiples of 16 rows, `_iq4xs_decode_kernel` for up to
    DECODE_ROWS rows, tinygrad's path otherwise."""
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    tokens = int(prod(x.shape[:-1]))
    if tokens <= DECODE_ROWS and tokens % 16 and layer.groups == 1 and layer.shard_axis is None:
        return _iq4xs_decode(layer, x, tokens, in_features, out_features)
    overrides = GROUPS_PER_STEP is not None or PACKED is not None or LDS_BYTES is not None or TILES is not None
    above = SERVES_ROWS is not None and tokens > SERVES_ROWS and not overrides
    if tokens % 16 or out_features % 16 or in_features % 256 or above:
        return fallback(layer, x)
    cdna = _is_cdna(x.device)
    kgroups, lanes = (4, 64) if cdna else (2 if _is_rdna4(x.device) else 1, 32)
    groups, packed = _config(tokens, in_features)
    rows = valid_rows()
    fxn = functools.partial(
        _iq4xs_wmma_kernel,
        out_features=out_features,
        in_features=in_features,
        kgroups=kgroups,
        lanes=lanes,
        groups_per_step=groups,
        packed=packed,
        no_weights=NO_WEIGHTS,
        lds_bytes=RESERVE_LDS if LDS_BYTES is None else LDS_BYTES,
        tiles=TILES,
        rows=rows,
    )
    out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
    xs = x.cast(dtypes.float16).contiguous()  # exactly tinygrad's form, see q8.py
    table = iq4_half_lut(x.device)
    result = Tensor.custom_kernel(out, layer.weight, bind_rows(xs, rows), table, fxn=fxn)[0].reshape(
        *x.shape[:-1], out_features
    )
    return result if layer.bias is None else result + layer.bias
