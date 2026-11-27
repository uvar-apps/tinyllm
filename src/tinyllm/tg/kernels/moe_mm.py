"""Grouped expert layers for prefill on the matrix cores (gfx11): each expert's weights are read once per tile of its
rows, instead of once per (token, expert) pair.

tinygrad's expert kernels (`tinygrad.llm.kernels.amd_gguf.moe_linear`, and `tinyllm.tg.kernels.moe_small` for
the types it lacks) compute one output row per (token, expert) pair, as decoding does, and read the pair's expert
weights for every row. A 128-token prefill chunk of Qwen3.8-Flash-Next `UD-Q4_K_XL` spent 574 of its 717 ms there
(docs/04-qwen3.8-flash-next.md, section 14). Here these kernels replace them for prefill:

  route    one workgroup, one thread per expert: each thread counts the pairs routed to its expert (16 pairs a
           load), prefix sums over the experts give each expert its place, and the thread writes its pairs there
           in order. Each expert's rows are cut into tiles of up to 128 rows (`WIDE_TILES`), into one table of
           (expert, first row, rows). The fused gate/up and the down projection of a block share one routing
           (`moe_forward`).
  block    one workgroup per (tile, 64 outputs), each wave 16 outputs of one weight matrix for all the tile's rows
           (`_block_kernel`): each weight is decoded once per tile, half a fragment per lane, the other half from
           the partner lane; weights load a block of 256 inputs at a time, a block ahead; activations are staged
           in local memory. The gate/up pair is two of these: the gate into a scratch buffer, then the up
           projection with SwiGLU in its stores.
  grouped  the earlier kernel, one per tile size (`WIDE_TILES` off, or experts of at most SMALL_MAX rows):
           `tinyllm.tg.kernels.wmma`'s loop, tiles of 64 rows and one of 16, 32, 48 or 64 for an expert's rest,
           gate and up fused in one kernel.

The grid has room for the most tiles any routing can need; a workgroup past the routing's tiles runs no step and
stores nothing. Real routing is skewed: in 2,048 tokens of code a layer's busiest expert gets up to 1,400 rows and
about 210 of the 512 get 16 or fewer.

The loop count of a workgroup is read from memory, which the pinned tinygrad's cost estimates cannot evaluate,
so the kernels carry their own (`Estimates`; they only feed statistics).

Padding pairs of a symbolic chunk (expert -1) are not routed: their rows are left unwritten and sliced off.

The dequantizers give the matrix cores 16 weights per lane in fp16, as tinygrad's matrix-core kernels for dense
layers do (`_q5_linear_f16_wmma_kernel`, `_quant_linear_f16_wmma_kernel`), with the same arithmetic where tinygrad
has the type. Activations are rounded to fp16. Decoding and the speculative verify keep the one-row kernels
(`MIN_TOKENS`).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.llm import model as llm_model
from tinygrad.llm.kernels.amd import _amd_load, _is_cdna, _is_rdna4, amd_custom_kernels_supported
from tinygrad.llm.kernels.amd_gguf import (
    IQ2_S,
    IQ3_XXS,
    IQ4_NL,
    IQ4_XS,
    IQ4NL_LUT,
    Q4_K,
    Q5_K,
    Q8_0,
    QUANT_NAMES,
    QUANT_SIZES,
    _half,
    _iq4_scale,
    _iq_grid,
    _load_byte,
    _load_u32,
    _nibble_lut,
    _q5_scales,
    _quant_word,
    find_packed,
)
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import AxisType, KernelInfo

from tinyllm.tg.kernels import moe_small
from tinyllm.tg.kernels.moe_small import Q2_0, Q5_0, Q5_1
from tinyllm.tg.kernels.wmma import (
    FP16_1024,
    byte_pairs,
    center_affine,
    center_scale,
    magic_pairs,
    splat_half,
    wmma_linear,
)

# fewer tokens in a call stay on the one-row-per-pair kernels (decoding, the speculative verify of a few drafts); a
# copied block's 16-row verify reads each expert once here: 158 -> 135 ms a cycle
MIN_TOKENS = 16
ROUTE_VECTOR = 16  # pairs the route kernel reads per load step
BIG, SMALL = 64, 16  # the largest tile (4 matrix tiles share each weight fragment) and the row granularity
WIDE_TILES = True  # experts with more than SMALL_MAX rows run in tiles of up to WIDE rows (`_block_kernel`)
WIDE = 128  # rows of a wide tile: 8 matrix tiles share each weight fragment
SMALL_MAX = 0  # experts with at most this many rows keep the per-wave kernel's tiles of 16, 32 or 48 rows
WIDE_WAVES = 4  # waves of a `_block_kernel` workgroup, 16 outputs each
WIDE_PAD = 8  # halves of padding per local-memory row of a wide tile's activations
WIDE_ACT_DEPTH = 2  # steps ahead a step's activations are loaded
HALF_DOWN = True  # the down projection's outputs in fp16, which the weighted sum over the experts reads
WIDE_DIAG = ""  # diagnostics, wrong results: "weights" (constant fragments), "inputs" (no activation loads), "both",
# "cached" (`_block_kernel`: every tile reads expert 0's weights, which stay in the caches)
WAVES, OUTPUT_TILES = 2, 1  # waves per workgroup and 16-output tiles per wave (per matrix when fused)
SHARED = False  # stage each step's gathered activations in local memory once per workgroup (wmma_linear)
FUSE_GATE_UP = True  # gate and up projections in one kernel, SwiGLU in its stores (`moe_forward`)
PREFETCH = False  # weights loaded a step ahead into registers (wmma_linear(prefetch=...))
GROUPS_PER_STEP = 4  # 128 inputs a step: the small tiles read more of each weight block at once (2: -1% wall clock)
# storage words per 256 weights, and the word type the kernels index
TYPE_WORDS = {
    Q4_K: QUANT_SIZES[Q4_K] // 4,
    Q5_K: QUANT_SIZES[Q5_K] // 4,
    Q8_0: 8 * 17,
    Q5_1: 8 * 6,
    IQ4_XS: QUANT_SIZES[IQ4_XS] // 4,  # u32 words
    IQ3_XXS: QUANT_SIZES[IQ3_XXS] // 2,  # halfwords from here on
    IQ2_S: QUANT_SIZES[IQ2_S] // 2,
    IQ4_NL: 8 * 9,
    Q5_0: 8 * 11,
    Q2_0: 4 * 9,
}
NAMES = {**QUANT_NAMES, Q5_1: "q5_1", Q5_0: "q5_0", Q2_0: "q2_0"}
GRID_TYPES = (IQ3_XXS, IQ2_S)  # their codebooks go to local memory
BLOCKS_256 = (Q4_K, Q5_K, IQ4_XS, IQ3_XXS, IQ2_S)  # 256-weight super-blocks: rows of a multiple of 256 inputs
PACKED = True  # fp16 conversion two weights at a time (wmma.center_affine); rounds a K-quant weight twice
Dequant = Callable[[UOp, UOp, int], tuple[UOp, ...] | UOp]


def _ireg(value: int) -> UOp:
    ret = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    return ret.after(ret[0].store(UOp.const(value, dtypes.int32)))


@functools.cache
def _route_kernel(
    rows: UOp, *outs: UOp, experts: int, most_full: int, big: int = BIG, small: int = 3, wide: bool = False
) -> UOp:
    """Pairs sorted by expert (`rows`), and a tile table per class. Class 0 holds tiles of `big` rows: an expert's
    whole groups of `big` rows, and with `wide` all the rows of an expert with more than `small` * 16, else a rest
    of more than `small` * 16 rows. The `small` other classes hold one tile of 48, 32 or 16 rows for the rest. `outs`
    holds (expert, first row, rows) of each class, then each class's total; `most_full` bounds one expert's class-0
    tiles."""
    *tables, sel = outs
    classes = len(tables) // 4
    (P,) = sel.shape
    one, zero = UOp.const(1, dtypes.int32), UOp.const(0, dtypes.int32)
    g, e = UOp.range(1, 0, AxisType.GLOBAL), UOp.range(experts, 1, AxisType.LOCAL)
    mine_id = e.cast(dtypes.int32)

    assert P % ROUTE_VECTOR == 0, "route() pads the pairs to whole vectors"

    def routed(p: UOp) -> list[UOp]:  # whether each of the pairs p * ROUTE_VECTOR .. is this thread's expert
        loads = [_amd_load(sel[p * ROUTE_VECTOR + 4 * k], 4) for k in range(ROUTE_VECTOR // 4)]
        return [v[i].eq(mine_id) for v in loads for i in range(4)]  # padding pairs (-1) match no expert

    def ones(mine: list[UOp]) -> UOp:
        return sum((m.where(one, zero) for m in mine[1:]), mine[0].where(one, zero))

    cnt = _ireg(0)
    p = UOp.range(P // ROUTE_VECTOR, 100)
    cnt = cnt.after(cnt[0].store(cnt.after(p)[0].load() + ones(routed(p))).end(p))
    count = cnt[0].load()
    full = count // big
    rest = count - full * big
    cls = (rest + (SMALL - 1)) // SMALL  # 16-row tiles the rest needs
    # whether the rest goes in a big tile (with `wide`, an expert has big tiles only or one small tile)
    measure = count if wide else rest
    in_big, in_small = measure > small * SMALL, measure <= small * SMALL
    # tiles per class, the big class first
    tiles = [full + (in_big & (rest > 0)).where(one, zero)]
    tiles += [(in_small & cls.eq(c)).where(one, zero) for c in range(small, 0, -1)]
    assert len(tiles) == classes
    # inclusive prefix sums over the experts of the row and tile counts (Hillis-Steele, double-buffered)
    values = (count, *tiles)
    scan = UOp.alloc((2, len(values), experts), dtypes.int32, addrspace=AddrSpace.LOCAL)
    buf = scan.after(*(scan[0, i, e].store(v) for i, v in enumerate(values)))
    q, off = 0, 1
    while off < experts:
        prev, has = (e - off).maximum(0), e >= off
        sums = [buf[q, i, e].load() + has.where(buf[q, i, prev].load(), zero) for i in range(len(values))]
        buf = scan.after(*(scan[1 - q, i, e].store(v) for i, v in enumerate(sums)))
        q, off = 1 - q, off * 2
    firsts = [buf[q, i, e].load() - v for i, v in enumerate(values)]
    first_row = firsts[0]

    written = _ireg(0)
    p2 = UOp.range(P // ROUTE_VECTOR, 101)
    taken = written.after(p2)[0].load()
    mine = routed(p2)
    fills, before = [], taken
    for i, m in enumerate(mine):  # this thread's pairs keep their order
        place = (first_row + before).cast(dtypes.weakint)
        fills.append(rows[place.valid(m)].store((p2 * ROUTE_VECTOR + i).cast(dtypes.int32)))
        before = before + m.where(one, zero)
    fill = UOp.group(*fills, written[0].store(before)).end(p2)
    groups = [fill]
    j = UOp.range(most_full, 102)
    jj = j.cast(dtypes.int32)
    slot = (firsts[1] + jj).cast(dtypes.weakint).valid(jj < tiles[0])
    expert_t, start_t, count_t = tables[0], tables[1], tables[2]
    groups.append(
        UOp.group(
            expert_t[slot].store(mine_id),
            start_t[slot].store(first_row + jj * big),
            count_t[slot].store((count - jj * big).minimum(big)),
        ).end(j)
    )
    for c in range(1, classes):  # the smaller rest tiles: at most one per expert
        slot_c = firsts[1 + c].cast(dtypes.weakint).valid(tiles[c].eq(1))
        expert_t, start_t, count_t = tables[4 * c], tables[4 * c + 1], tables[4 * c + 2]
        groups.append(
            UOp.group(
                expert_t[slot_c].store(mine_id),
                start_t[slot_c].store(first_row + full * big),
                count_t[slot_c].store(rest),
            )
        )
    last = UOp.const(0, dtypes.weakint).valid(e.eq(experts - 1))
    groups.append(UOp.group(*(tables[4 * c + 3][last].store(buf[q, 1 + c, e].load()) for c in range(classes))))
    return UOp.group(*groups).end(e, g).sink(arg=KernelInfo(name="moe_route", opts_to_apply=()))


def _int8_packed(words: list[UOp], scale: UOp) -> UOp:
    """16 signed byte codes (4 words) times `scale`, as 16 fp16: each pair of bytes under 1024's exponent with the
    sign bit flipped (1024 + 128 + code exactly), then one packed multiply-add per pair (as q8.py does)."""
    s2 = splat_half(scale.cast(dtypes.half))
    magic = (FP16_1024 | FP16_1024 << 16) | 0x00800080
    return UOp.stack(*(center_scale(p ^ magic, s2, 1152) for w in words for p in byte_pairs(w))).bitcast(dtypes.half)


def _dequantizer(ggml_type: int, raw: UOp, grid: UOp | None = None) -> Dequant:
    """dequant(base, subgroup, half): this lane's 16 weights (RDNA3: one output row, 16 consecutive inputs) of half
    `half` of group `subgroup` of the 256-weight block at word `base`, in fp16."""
    if ggml_type in (Q4_K, Q5_K):

        def kquant(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
            d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
            qs = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2) * 8 + half * 4
            words = []
            for i in range(4):
                word = raw[qs + i] >> ((subgroup & 1) * 4).cast(dtypes.uint32) & 0x0F0F0F0F
                if ggml_type == Q5_K:
                    word = word | (((raw[base + 4 + half * 4 + i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4)
                words.append(word)
            if PACKED:  # two weights per instruction: (1024 + q - 1024) * d * sc - dmin * m
                ds2 = splat_half((d * scale).cast(dtypes.half))
                dm2 = splat_half((-(dmin * minimum)).cast(dtypes.half))
                packed = [center_affine(pair, ds2, dm2, 1024) for w in words for pair in magic_pairs(w)]
                return UOp.stack(*packed).bitcast(dtypes.half)
            return tuple(
                ((w >> (b * 8) & 255).float() * d * scale - dmin * minimum).cast(dtypes.float16)
                for w in words
                for b in range(4)
            )

        return kquant
    if ggml_type == Q8_0:  # halfwords: d, then 32 int8 codes

        def q8(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...]:
            block = base + subgroup * 17
            scale = _half(raw[block])
            words = [_load_u32(raw, block, 2 + (half * 4 + i) * 4, stream=True) for i in range(4)]
            return tuple(
                ((w >> (b * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() * scale).cast(dtypes.half)
                for w in words
                for b in range(4)
            )

        return q8
    if ggml_type == Q5_1:  # u32 words: d | m, qh, then 4 words of nibbles; w = d * q + m

        def q5_1(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
            block = base + subgroup * 6
            dm, qh = raw[block], raw[block + 1].cast(dtypes.uint32)
            d, m = _half(dm & 0xFFFF), _half(dm >> 16)
            qs = [raw[block + 2 + w].cast(dtypes.uint32) for w in range(4)]
            if PACKED:  # four 5-bit codes a word: the nibbles, and the four qh bits spread to bit 4 of each byte
                words = [
                    ((qs[w] >> (4 * half)) & 0x0F0F0F0F)
                    | (((((qh >> (16 * half + 4 * w)) & 15) * 0x00204081) & 0x01010101) << 4)
                    for w in range(4)
                ]
                d2, m2 = splat_half(dm & 0xFFFF), splat_half(dm >> 16)
                packed = [center_affine(pair, d2, m2, 1024) for w in words for pair in magic_pairs(w)]
                return UOp.stack(*packed).bitcast(dtypes.half)
            codes = [
                ((qs[i // 4] >> (8 * (i % 4) + 4 * half)) & 15) | (((qh >> (16 * half + i)) & 1) << 4)
                for i in range(16)
            ]
            return tuple((code.float() * d + m).cast(dtypes.half) for code in codes)

        return q5_1
    if ggml_type == IQ4_XS and PACKED:  # 136-byte blocks are only 8-byte aligned: vector loads of codes and header

        def iq4xs(base: UOp, subgroup: UOp, half: int) -> UOp:
            head = _amd_load(raw[base], 2)  # d | scales_h << 16, then the low scale nibbles
            ls = ((head[1] >> (subgroup * 4).cast(dtypes.uint32)) & 15) | (
                ((head[0] >> (subgroup * 2 + 16).cast(dtypes.uint32)) & 3) << 4
            )
            scale = _half(head[0] & 0xFFFF) * (ls.cast(dtypes.int32) - 32).float()
            codes = _amd_load(raw[base + 2 + subgroup * 4], 4)
            return _int8_packed([_nibble_lut(codes[i], 4 * half, IQ4NL_LUT) for i in range(4)], scale)

        return iq4xs
    if ggml_type in (IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_S):  # tinygrad's codes and scales (amd_gguf), times the scale

        def iq(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
            block = base + subgroup * 9 if ggml_type == IQ4_NL else base
            if ggml_type == IQ4_XS:
                scale = _iq4_scale(raw, base, subgroup)
            elif ggml_type == IQ4_NL:
                scale = _half(raw[block])
            elif ggml_type == IQ3_XXS:
                scale = _half(raw[base]) * ((_load_u32(raw, base, 66 + subgroup * 4) >> 28).float() + 0.5) * 0.5
            else:
                sc = _load_byte(raw, base, 74 + subgroup)
                scale = _half(raw[base]) * (((sc >> (half * 4)) & 15).float() + 0.5) * 0.25
            words = [_quant_word(raw, block, subgroup, half * 4 + i, ggml_type, grid) for i in range(4)]
            if PACKED:  # signed byte codes, two at a time: (1024 + 128 + code - 1152) * scale
                return _int8_packed(words, scale)
            return tuple(
                ((w >> (b * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() * scale).cast(dtypes.half)
                for w in words
                for b in range(4)
            )

        return iq
    if ggml_type == Q5_0:  # halfwords: d, qh (bytes 2..5), 16 bytes of nibbles; w = d * (q - 16)

        def q5_0(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
            block = base + subgroup * 11
            d = _half(raw[block])
            qh, qs = _load_u32(raw, block, 2), [_load_u32(raw, block, 6 + 4 * w) for w in range(4)]
            if PACKED:  # four 5-bit codes a word, as Q5_1: (1024 + q - 1040) * d
                words = [
                    ((qs[w] >> (4 * half)) & 0x0F0F0F0F)
                    | (((((qh >> (16 * half + 4 * w)) & 15) * 0x00204081) & 0x01010101) << 4)
                    for w in range(4)
                ]
                d2 = splat_half(d.cast(dtypes.half))
                return UOp.stack(*(center_scale(p, d2, 1040) for w in words for p in magic_pairs(w))).bitcast(
                    dtypes.half
                )
            codes = [
                ((qs[i // 4] >> (8 * (i % 4) + 4 * half)) & 15) | (((qh >> (16 * half + i)) & 1) << 4)
                for i in range(16)
            ]
            return tuple(((code.cast(dtypes.int32) - 16).float() * d).cast(dtypes.half) for code in codes)

        return q5_0
    if ggml_type == Q2_0:  # 64 weights in 9 halfwords: d, then four 2-bit codes a byte; w = d * (q - 1)

        def q2_0(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
            block = base + (subgroup // 2) * 9
            d = _half(raw[block])
            word = _load_u32(raw, block, 2 + (subgroup % 2) * 8 + 4 * half)
            if PACKED:  # four 2-bit codes spread to a word's bytes: (1024 + q - 1025) * d
                words = []
                for k in range(4):
                    x = (word >> (8 * k)) & 255
                    words.append((x & 3) | ((x >> 2) & 3) << 8 | ((x >> 4) & 3) << 16 | ((x >> 6) & 3) << 24)
                d2 = splat_half(d.cast(dtypes.half))
                return UOp.stack(*(center_scale(p, d2, 1025) for w in words for p in magic_pairs(w))).bitcast(
                    dtypes.half
                )
            return tuple(
                ((((word >> (2 * i)) & 3).cast(dtypes.int32) - 1).float() * d).cast(dtypes.half) for i in range(16)
            )

        return q2_0
    raise ValueError(f"no grouped expert kernel for ggml type {ggml_type}")


def _fetcher(ggml_type: int, raw: UOp) -> tuple[Callable[..., list[UOp]], Callable[..., UOp]]:
    """The dequantizer split for `wmma_linear(prefetch=...)`: fetch(base, subgroup, half) -> raw words, loaded a step
    ahead into registers; convert(words, subgroup, half) -> the lane's 16 fp16 weights (packed conversion)."""
    if ggml_type in (Q4_K, Q5_K):
        q5 = ggml_type == Q5_K

        def fetch_k(base: UOp, subgroup: UOp, half: int) -> list[UOp]:
            qs = base + (12 if q5 else 4) + (subgroup // 2) * 8 + half * 4
            words = [raw[base + i] for i in range(4)] + [raw[qs + i] for i in range(4)]
            return words + ([raw[base + 4 + half * 4 + i] for i in range(4)] if q5 else [])

        def convert_k(words: list[UOp], subgroup: UOp, half: int) -> UOp:
            head, w1, w2, w3 = words[:4]
            sb = ((subgroup & 3) * 8).cast(dtypes.uint32)
            byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
            scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4)).float()
            minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4)).float()
            d, dmin = _half(head & 0xFFFF), _half(head >> 16)
            shift = ((subgroup & 1) * 4).cast(dtypes.uint32)
            codes = []
            for i in range(4):
                code = (words[4 + i] >> shift) & 0x0F0F0F0F
                if q5:
                    code = code | (((words[8 + i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4)
                codes.append(code)
            ds2, dm2 = splat_half((d * scale).cast(dtypes.half)), splat_half((-(dmin * minimum)).cast(dtypes.half))
            packed = [center_affine(pair, ds2, dm2, 1024) for c in codes for pair in magic_pairs(c)]
            return UOp.stack(*packed).bitcast(dtypes.half)

        return fetch_k, convert_k
    if ggml_type == Q5_1:

        def fetch_5(base: UOp, subgroup: UOp, half: int) -> list[UOp]:
            return [raw[base + subgroup * 6 + i] for i in range(6)]

        def convert_5(words: list[UOp], subgroup: UOp, half: int) -> UOp:
            dm, qh, qs = words[0], words[1], words[2:]
            codes = [
                ((qs[w] >> (4 * half)) & 0x0F0F0F0F)
                | (((((qh >> (16 * half + 4 * w)) & 15) * 0x00204081) & 0x01010101) << 4)
                for w in range(4)
            ]
            d2, m2 = splat_half(dm & 0xFFFF), splat_half(dm >> 16)
            return UOp.stack(*(center_affine(p, d2, m2, 1024) for c in codes for p in magic_pairs(c))).bitcast(
                dtypes.half
            )

        return fetch_5, convert_5
    if ggml_type == Q8_0:

        def fetch_8(base: UOp, subgroup: UOp, half: int) -> list[UOp]:
            block = base + subgroup * 17
            return [raw[block].cast(dtypes.uint32)] + [_load_u32(raw, block, 2 + (half * 4 + i) * 4) for i in range(4)]

        def convert_8(words: list[UOp], subgroup: UOp, half: int) -> UOp:
            scale = words[0].cast(dtypes.uint16).bitcast(dtypes.half).float()
            values = (
                ((w >> (b * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() * scale).cast(dtypes.half)
                for w in words[1:]
                for b in range(4)
            )
            return UOp.stack(*values)

        return fetch_8, convert_8
    raise ValueError(f"no grouped expert kernel for ggml type {ggml_type}")


@functools.cache
def _grouped_kernel(
    out: UOp,
    raw: UOp,
    x: UOp,
    rows: UOp,
    tile_expert: UOp,
    tile_start: UOp,
    total: UOp,
    tile_count: UOp,
    *extra: UOp,
    out_features: int,
    in_features: int,
    ggml_type: int,
    x_share: int,
    tile: int,
    fused: bool = False,
) -> UOp:
    """out[pair] = x[pair's row] @ W[pair's expert].T for the pairs of the tile table; with `raw_up` (a gate/up
    pair of the same type) out[pair] = SwiGLU(gate, up) in fp16, the down projection's input."""
    raw_up = extra[0] if fused else None
    grid_in = extra[-1] if ggml_type in GRID_TYPES else None
    type_words = TYPE_WORDS[ggml_type]
    expert_words = out_features * in_features * type_words // 256
    slots, P, experts = int(tile_expert.shape[0]), int(rows.shape[0]), int(raw.shape[0]) // expert_words
    if out_features % (16 * OUTPUT_TILES * WAVES):
        raise ValueError(f"{out_features} outputs are not whole blocks of {16 * OUTPUT_TILES * WAVES}")
    tile_slot = UOp.range(slots, 0, AxisType.GLOBAL)
    output_block = UOp.range(out_features // (16 * OUTPUT_TILES * WAVES), 1, AxisType.GLOBAL)
    lane, wave = UOp.range(32, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
    col, lane_group = lane % 16, lane // 16

    def loaded(buf: UOp, i: UOp | int, lo: int, hi: int) -> UOp:
        # loaded indices clamped to what they can be: with known bounds the address arithmetic simplifies and
        # the activation loads become vector loads
        return buf[i].load().cast(dtypes.weakint).maximum(lo).minimum(hi)

    live = tile_slot < loaded(total, 0, 0, slots)  # workgroups past the routing's tiles do nothing
    t = tile_slot.minimum(slots - 1)
    expert = loaded(tile_expert, t, 0, experts - 1)
    start = loaded(tile_start, t, 0, P - 1)
    count = loaded(tile_count, t, 1, tile) if tile_count is not None else UOp.const(tile, dtypes.weakint)

    def pair(r: int | UOp) -> UOp:
        return loaded(rows, (start + (count - 1).minimum(r)).minimum(P - 1), 0, P - 1)

    outputs = tuple((output_block * WAVES + wave) * (16 * OUTPUT_TILES) + o * 16 + col for o in range(OUTPUT_TILES))
    if raw_up is not None:
        outputs = outputs + outputs  # the same outputs of the gate, then of the up matrix
    inputs = tuple(pair(s * 16 + col) // x_share for s in range(tile // 16))
    tokens = tuple(
        tuple(pair(r).valid(live & (r < count)) for r in (s * 16 + lane_group * 8 + i for i in range(8)))
        for s in range(tile // 16)
    )
    layout = (WAVES, tile_slot, output_block, lane, wave, lane_group, outputs, inputs, tokens)

    grid = None
    if grid_in is not None:  # the codebook in local memory, copied by every thread
        threads, n = WAVES * 32, int(grid_in.shape[0])
        tid = wave * 32 + lane
        grid = UOp.alloc((n,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
        grid = grid.after(
            *(grid[tid + i * threads].store(grid_in[tid + i * threads].load()) for i in range(n // threads))
        )

    def expert_dequant(raw: UOp) -> Dequant:
        dequant = _dequantizer(ggml_type, raw, grid)
        return lambda base, subgroup, half: dequant(base + expert * expert_words, subgroup, half)

    dequants: Dequant | tuple[Dequant, ...] = expert_dequant(raw)
    if raw_up is not None:
        dequants = (expert_dequant(raw),) * OUTPUT_TILES + (expert_dequant(raw_up),) * OUTPUT_TILES

    groups = GROUPS_PER_STEP if (in_features // 32) % GROUPS_PER_STEP == 0 else 1
    # the most a launch can do: every slot a whole tile
    flops = 2 * slots * tile * out_features * in_features
    weight_bytes = experts * expert_words * raw.dtype.itemsize
    estimates = Estimates(ops=flops, lds=weight_bytes, mem=weight_bytes + slots * tile * out_features * 4)
    return wmma_linear(
        out,
        x,
        out_features,
        in_features,
        type_words,
        layout,
        dequants,
        f"moe_mm_{NAMES[ggml_type]}{'_gate_up' if raw_up is not None else ''}_t{tile}",
        1,
        32,
        groups_per_step=groups,
        x_rows=int(x.shape[0]),
        steps=live.where(in_features // 32 // groups, 0),
        estimates=estimates,
        share_inputs=SHARED,
        input_row=lambda r: pair(r) // x_share,
        prefetch=_prefetch(ggml_type, raw, raw_up, expert * expert_words) if PREFETCH else None,
        stores=_swiglu_stores if raw_up is not None else None,
    )


def _int8_pairs(words: list[UOp], scale: UOp) -> list[UOp]:
    """`_int8_packed` as a list of words: each holds two fp16 weights, signed byte code times `scale`."""
    s2 = splat_half(scale.cast(dtypes.half))
    magic = (FP16_1024 | FP16_1024 << 16) | 0x00800080
    return [center_scale(p ^ magic, s2, 1152) for w in words for p in byte_pairs(w)]


def _spread4(nibble: UOp) -> UOp:
    """Bits 0..3 of `nibble` moved to bit 4 of bytes 0..3 (the fifth code bit of four Q5 weights)."""
    return (((nibble & 15) * 0x00204081) & 0x01010101) << 4


Fetch = Callable[[UOp, UOp, UOp], list[UOp]]
Convert = Callable[[list[UOp], UOp, UOp, UOp], list[list[UOp]]]


def _half_codec(ggml_type: int, raw: UOp, grid: UOp | None = None) -> tuple[Fetch, Convert]:
    """A lane's share of one step (32 inputs) of a weight row, for `_block_kernel`, as two functions:

      fetch(base, subgroup, part) -> the raw 32-bit words the lane needs of group `subgroup` of the 256-weight
        block at word `base`, loaded a step ahead into registers;
      convert(words, base, subgroup, part) -> for each half of the group (16 inputs, one matrix fragment), 4 words
        of two fp16 weights each: inputs 8 * part .. 8 * part + 7 of that half.

    The arithmetic is `_dequantizer`'s. `part` may differ per lane. The codebook types (IQ3_XXS, IQ2_S) load in
    `convert`, since their codebook indices are themselves loaded words."""
    u32 = dtypes.uint32
    if ggml_type in (Q4_K, Q5_K):
        q5 = ggml_type == Q5_K

        def fetch_k(base: UOp, subgroup: UOp, part: UOp) -> list[UOp]:
            head = _amd_load(raw[base], 4)  # d | dmin, then the 12 bytes of 6-bit scales and minimums
            words = [head[i] for i in range(4)]
            for half in range(2):
                codes = _amd_load(raw[base + (12 if q5 else 4) + (subgroup // 2) * 8 + half * 4 + part * 2], 2)
                words += [codes[0], codes[1]]
                if q5:
                    high = _amd_load(raw[base + 4 + half * 4 + part * 2], 2)
                    words += [high[0], high[1]]
            return words

        def convert_k(words: list[UOp], base: UOp, subgroup: UOp, part: UOp) -> list[list[UOp]]:
            sb = ((subgroup & 3) * 8).cast(u32)
            byte1, byte2, byte3 = (words[1] >> sb) & 255, (words[2] >> sb) & 255, (words[3] >> sb) & 255
            low = subgroup < 4
            scale = low.where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4)).float()
            minimum = low.where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4)).float()
            d, dmin = _half(words[0] & 0xFFFF), _half(words[0] >> 16)
            ds2 = splat_half((d * scale).cast(dtypes.half))
            dm2 = splat_half((-(dmin * minimum)).cast(dtypes.half))
            shift, per = ((subgroup & 1) * 4).cast(u32), (4 if q5 else 2)
            halves = []
            for half in range(2):
                w = words[4 + half * per : 4 + (half + 1) * per]
                codes = [(w[i] >> shift) & 0x0F0F0F0F for i in range(2)]
                if q5:
                    codes = [c | (((w[2 + i] >> subgroup.cast(u32)) & 0x01010101) << 4) for i, c in enumerate(codes)]
                halves.append([center_affine(p, ds2, dm2, 1024) for c in codes for p in magic_pairs(c)])
            return halves

        return fetch_k, convert_k
    if ggml_type == Q8_0:  # halfwords: d, then 32 int8 codes

        def fetch_8(base: UOp, subgroup: UOp, part: UOp) -> list[UOp]:
            block = base + subgroup * 17
            codes = [_load_u32(raw, block, 2 + half * 16 + part * 8 + 4 * i) for half in range(2) for i in range(2)]
            return [raw[block].cast(u32), *codes]

        def convert_8(words: list[UOp], base: UOp, subgroup: UOp, part: UOp) -> list[list[UOp]]:
            return [_int8_pairs(words[1 + 2 * half : 3 + 2 * half], _half(words[0])) for half in range(2)]

        return fetch_8, convert_8
    if ggml_type in (Q5_1, Q5_0):  # Q5_1: u32 words d | m, qh, 4 words of nibbles; Q5_0: halfwords d, qh, nibbles
        q51 = ggml_type == Q5_1

        def fetch_5(base: UOp, subgroup: UOp, part: UOp) -> list[UOp]:
            if q51:
                block = base + subgroup * 6
                head, qs = _amd_load(raw[block], 2), _amd_load(raw[block + 2 + 2 * part], 2)
                return [head[0], head[1], qs[0], qs[1]]
            block = base + subgroup * 11
            codes = [_load_u32(raw, block, 6 + 8 * part + 4 * i) for i in range(2)]
            return [raw[block].cast(u32), _load_u32(raw, block, 2), *codes]

        def convert_5(words: list[UOp], base: UOp, subgroup: UOp, part: UOp) -> list[list[UOp]]:
            dm, qh, qs = words[0], words[1], words[2:4]
            halves = []
            for half in range(2):
                shift = (16 * half + 8 * part).cast(u32)
                codes = [((c >> (4 * half)) & 0x0F0F0F0F) | _spread4(qh >> (shift + 4 * i)) for i, c in enumerate(qs)]
                if q51:
                    d2, m2 = splat_half(dm & 0xFFFF), splat_half(dm >> 16)
                    halves.append([center_affine(p, d2, m2, 1024) for c in codes for p in magic_pairs(c)])
                else:
                    d2 = splat_half(dm & 0xFFFF)
                    halves.append([center_scale(p, d2, 1040) for c in codes for p in magic_pairs(c)])
            return halves

        return fetch_5, convert_5
    if ggml_type == IQ4_NL:  # halfwords: d, 16 bytes of nibbles

        def fetch_nl(base: UOp, subgroup: UOp, part: UOp) -> list[UOp]:
            block = base + subgroup * 9
            return [raw[block].cast(u32), *(_load_u32(raw, block, 2 + 8 * part + 4 * i) for i in range(2))]

        def convert_nl(words: list[UOp], base: UOp, subgroup: UOp, part: UOp) -> list[list[UOp]]:
            scale = _half(words[0])
            return [_int8_pairs([_nibble_lut(p, 4 * half, IQ4NL_LUT) for p in words[1:3]], scale) for half in range(2)]

        return fetch_nl, convert_nl
    if ggml_type == IQ4_XS:  # u32 words: d | scales_h << 16, scales_l, then 4 words of nibbles per group

        def fetch_xs(base: UOp, subgroup: UOp, part: UOp) -> list[UOp]:
            head, codes = _amd_load(raw[base], 2), _amd_load(raw[base + 2 + subgroup * 4 + 2 * part], 2)
            return [head[0], head[1], codes[0], codes[1]]

        def convert_xs(words: list[UOp], base: UOp, subgroup: UOp, part: UOp) -> list[list[UOp]]:
            h0, h1 = words[0], words[1]
            ls = ((h1 >> (subgroup * 4).cast(u32)) & 15) | (((h0 >> (subgroup * 2 + 16).cast(u32)) & 3) << 4)
            scale = _half(h0 & 0xFFFF) * (ls.cast(dtypes.int32) - 32).float()
            return [_int8_pairs([_nibble_lut(c, 4 * half, IQ4NL_LUT) for c in words[2:4]], scale) for half in range(2)]

        return fetch_xs, convert_xs
    if ggml_type == Q2_0:  # 64 weights in 9 halfwords: d, then four 2-bit codes a byte

        def fetch_2(base: UOp, subgroup: UOp, part: UOp) -> list[UOp]:
            block = base + (subgroup // 2) * 9
            codes = [raw[block + 1 + (subgroup % 2) * 4 + half * 2 + part].cast(u32) for half in range(2)]  # 8 each
            return [raw[block].cast(u32), *codes]

        def convert_2(words: list[UOp], base: UOp, subgroup: UOp, part: UOp) -> list[list[UOp]]:
            d2 = splat_half(words[0] & 0xFFFF)
            halves = []
            for codes in words[1:3]:
                spread = []
                for k in range(2):
                    x = (codes >> (8 * k)) & 255
                    spread.append((x & 3) | ((x >> 2) & 3) << 8 | ((x >> 4) & 3) << 16 | ((x >> 6) & 3) << 24)
                halves.append([center_scale(p, d2, 1025) for w in spread for p in magic_pairs(w)])
            return halves

        return fetch_2, convert_2
    if ggml_type in (IQ3_XXS, IQ2_S):  # tinygrad's codebook words (amd_gguf._quant_word), times the scale

        def fetch_none(base: UOp, subgroup: UOp, part: UOp) -> list[UOp]:
            return []

        def convert_iq(words: list[UOp], base: UOp, subgroup: UOp, part: UOp) -> list[list[UOp]]:
            halves = []
            for half in range(2):
                if ggml_type == IQ3_XXS:
                    scale = _half(raw[base]) * ((_load_u32(raw, base, 66 + subgroup * 4) >> 28).float() + 0.5) * 0.5
                else:
                    sc = _load_byte(raw, base, 74 + subgroup)
                    scale = _half(raw[base]) * (((sc >> (half * 4)) & 15).float() + 0.5) * 0.25
                codes = [_quant_word(raw, base, subgroup, half * 4 + part * 2 + i, ggml_type, grid) for i in range(2)]
                halves.append(_int8_pairs(codes, scale))
            return halves

        return fetch_none, convert_iq
    raise ValueError(f"no wide expert kernel for ggml type {ggml_type}")


# v_permlanex16 with the identity selectors: lane i of each half-wave reads lane i of the other half
_PERMLANEX16 = "__builtin_amdgcn_permlanex16({0}, {0}, 0x76543210u, 0xfedcba98u, false, false)"
_WMMA = "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32"
# a workgroup barrier that orders local memory only: a fence for all memory waits for every outstanding load
_LDS_FENCE = (
    '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); __builtin_amdgcn_s_barrier(); '
    '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); '
)


def _ph(i: int) -> str:
    """A format placeholder for source i of a native call."""
    return "{" + str(i) + "}"


def _swap_halves(word: UOp) -> UOp:
    """The 32-bit `word` of lane l ^ 16."""
    return UOp.native(_PERMLANEX16, shape=(), dtype=dtypes.uint32, fmt=True).call(word.cast(dtypes.uint32), inline=True)


def _token(code: str, *srcs: UOp) -> UOp:
    """A C block with side effects, as an int that later blocks take as a source: tinygrad orders the blocks by
    these sources, and each block's own fences order its memory accesses."""
    uses = "".join(f"(void)({_ph(i)}); " for i in range(len(srcs)))
    return UOp.native("({{ " + uses + code + "0; }})", shape=(), dtype=dtypes.int32, fmt=True).call(*srcs)


def _lds_store(token: UOp, places: list[UOp], values: list[UOp], keep: list[UOp | None]) -> UOp:
    """Stores of 8 fp16 values to each local-memory place (where its `keep` is None or true), after `token`'s
    block."""
    code = "typedef _Float16 tl_h8 __attribute__((ext_vector_type(8), aligned(16))); "
    srcs = [token]
    for place, value, cond in zip(places, values, keep, strict=True):
        srcs += [place, value]
        store = f"*(tl_h8*)({_ph(len(srcs) - 2)}) = {_ph(len(srcs) - 1)}; "
        if cond is not None:
            srcs.append(cond)
            store = f"if ({_ph(len(srcs) - 1)}) {store}"
        code += store
    uses = f"(void)({_ph(0)}); "
    return UOp.native("({{ " + uses + code + "0; }})", shape=(), dtype=dtypes.int32, fmt=True).call(*srcs)


def _tile_step(
    afrags: list[list[UOp]], row: UOp, accs: list[UOp], run: UOp | None, token: UOp
) -> tuple[list[UOp], UOp]:
    """One step of one 16-row matrix tile, after `token`'s block: accs[m] + sum over k of afrags[k][m] @ B_k for each
    matrix m, where B_k is inputs 16k .. 16k + 15 of the tile's rows, read from local memory at `row` (this lane's
    row, the step's first input). The reads and matrix instructions are one block of C, skipped when `run` (the same
    for the whole wave) is false: the compiler cannot hoist the reads of every tile ahead of the matrix work, which
    would hold all their fragments in registers at once. Returns the new accs and the block's call."""
    n = len(accs)
    srcs = [*accs, *(a for frags in afrags for a in frags), row, token]
    body = ""
    for k in range(len(afrags)):
        body += f"tl_h16 tl_b{k} = *(tl_h16*)({_ph(len(srcs) - 2)} + {16 * k}); "
        body += "".join(f"tl_c{m} = {_WMMA}({_ph(n + k * n + m)}, tl_b{k}, tl_c{m}); " for m in range(n))
    if run is not None:
        srcs.append(run)
        body = f"if ({_ph(len(srcs) - 1)}) {{{{ {body}}}}} "
    result = "tl_c0" if n == 1 else f"__builtin_shufflevector(tl_c0, tl_c1, {', '.join(map(str, range(16)))})"
    code = (
        "({{ typedef _Float16 tl_h16 __attribute__((ext_vector_type(16), aligned(16))); "
        + f"(void)({_ph(len(srcs) - (2 if run is not None else 1))}); float8 "
        + ", ".join(f"tl_c{m} = {_ph(m)}" for m in range(n))
        + "; "
        + body
        + result
        + "; }})"
    )
    out = UOp.native(code, shape=(8 * n,), dtype=dtypes.float, fmt=True).call(*srcs)
    # lanes by subscript: tinygrad renders an index into a call's vector as pointer arithmetic
    lanes = [
        UOp.native(f"({{0}})[{i}]", shape=(), dtype=dtypes.float, fmt=True).call(out, inline=True) for i in range(8 * n)
    ]
    return [UOp.stack(*lanes[8 * m : 8 * m + 8]) for m in range(n)], lanes[0]


_OPAQUE_ZERO = '({{ (void)({0}); int tl_z = 0; asm volatile("" : "+v"(tl_z)); tl_z; }})'


def _opaque_zero(token: UOp) -> UOp:
    """0, known only once `token`'s block has run: an index offset that keeps a load after that block."""
    return UOp.native(_OPAQUE_ZERO, shape=(), dtype=dtypes.int32, fmt=True).call(token).cast(dtypes.weakint)


@functools.cache
def _block_kernel(
    out: UOp,
    raw: UOp,
    x: UOp,
    rows: UOp,
    tile_expert: UOp,
    tile_start: UOp,
    total: UOp,
    tile_count: UOp,
    *extra: UOp,
    out_features: int,
    in_features: int,
    ggml_type: int,
    x_share: int,
    swiglu: bool = False,
    half: bool = False,
    tile: int = WIDE,
) -> UOp:
    """`_grouped_kernel`'s product over tiles of up to `tile` rows: one workgroup per (tile, WIDE_WAVES * 16
    outputs), each wave 16 outputs of one weight matrix for all the tile's rows.

    The weights are the matrix instruction's A operand: lane l holds row l % 16 (an output). The instruction reads an
    even row from the low half-wave's lane and an odd row from the high half's, so each lane dequantizes 8 of a
    fragment's 16 inputs (`part`) and takes the other 8 from lane l ^ 16 (v_permlanex16). Each weight is decoded
    once per tile, for all its 8 matrix tiles; a matrix tile past the tile's rows issues no instruction
    (`_tile_step`).

    Each lane loads its share of a block of the wave's weight rows (256 inputs, or 4 groups of 32 for the 32-weight
    types when the inputs are not whole blocks of 256) in one go, a block ahead, and uses all of it over the
    block's steps: a cache line of weights is read once and used at once (loaded per 32 inputs, the same kernel was
    20-30% slower). The step loop is unrolled over the block, so a K-quant's group position is a constant.

    The activations are the B operand, staged in local memory 32 inputs (a step) at a time by all threads, loaded
    WIDE_ACT_DEPTH steps ahead into registers. Each load is issued after a barrier (`_opaque_zero`), so that only
    those steps' are held. The local memory writes, reads and barriers are C blocks (`_lds_store`, `_tile_step`,
    `_LDS_FENCE`) whose barriers fence local memory only: tinygrad's own barrier waits for every load in flight.

    The output fragment holds outputs 2v + l // 16 of token l % 16; lanes l and l ^ 16 exchange halves so that each
    stores 8 consecutive outputs. Without `swiglu` the kernel stores x @ W.T in fp32 (the down projection, or the
    gate of a gate/up pair, into a scratch buffer). With `swiglu`, extra[0] is that gate scratch and the kernel
    stores SwiGLU(gate, x @ W_up.T) in fp16. One matrix per wave keeps 64 accumulator registers; a gate/up pair in
    one wave needed 128 and could not also hold a block of weights."""
    gate_in = extra[0] if swiglu else None
    grid_in = extra[-1] if ggml_type in GRID_TYPES else None
    u32 = dtypes.uint32
    type_words = TYPE_WORDS[ggml_type]
    output_words = in_features * type_words // 256
    expert_words = out_features * output_words
    slots, P, experts = int(tile_expert.shape[0]), int(rows.shape[0]), int(raw.shape[0]) // expert_words
    waves, tiles16 = WIDE_WAVES, tile // 16
    BN, threads, per_row = 16 * waves, 32 * waves, 4  # outputs per workgroup; 8-input chunks of a row per step
    n_steps, chunks = in_features // 32, tile * per_row
    unroll = 8 if ggml_type in BLOCKS_256 else next(u for u in (8, 4, 2) if n_steps % u == 0)
    if out_features % BN or tile % 16 or n_steps % unroll:
        raise ValueError(f"{out_features} x {in_features}: not whole blocks of {BN} outputs and {unroll * 32} inputs")
    n_phases = n_steps // unroll
    x = x.reshape(int(x.shape[0]), in_features)
    tile_slot = UOp.range(slots, 0, AxisType.GLOBAL)
    output_block = UOp.range(out_features // BN, 1, AxisType.GLOBAL)
    wave, lane = UOp.range(waves, 3, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
    tid = wave * 32 + lane
    col, lane_group = lane % 16, lane // 16
    part = lane_group ^ (col & 1)  # 0 in the lane the instruction reads row `col` from

    def loaded(buf: UOp, i: UOp | int, lo: int, hi: int) -> UOp:
        return buf[i].load().cast(dtypes.weakint).maximum(lo).minimum(hi)

    live = tile_slot < loaded(total, 0, 0, slots)  # workgroups past the routing's tiles do nothing
    t = tile_slot.minimum(slots - 1)
    expert = loaded(tile_expert, t, 0, experts - 1)
    start = loaded(tile_start, t, 0, P - 1)
    count = loaded(tile_count, t, 1, tile)

    def pair(r: UOp | int) -> UOp:
        return loaded(rows, (start + (count - 1).minimum(r)).minimum(P - 1), 0, P - 1)

    grid = None
    if grid_in is not None:  # the codebook in local memory, copied by every thread
        n = int(grid_in.shape[0])
        grid = UOp.alloc((n,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
        grid = grid.after(
            *(grid[tid + i * threads].store(grid_in[tid + i * threads].load()) for i in range(n // threads))
        )
    fetch, convert = _half_codec(ggml_type, raw, grid)
    weight_expert = expert * 0 if WIDE_DIAG == "cached" else expert  # cached: every tile reads expert 0's weights
    row_base = weight_expert * expert_words + (output_block * BN + wave * 16 + col) * output_words

    def place_of(ph: UOp, j: int) -> tuple[UOp, UOp]:
        """(block word, group) of step j of block phase `ph`."""
        if ggml_type in BLOCKS_256:
            return row_base + ph * type_words, UOp.const(j, dtypes.weakint)
        s = ph * unroll + j
        return row_base + (s // 8) * type_words, s % 8

    def block_words(ph: UOp) -> tuple[list[UOp], list[list[int]]]:
        """The lane's words of block phase `ph`, each once, and which of them each step converts."""
        unique: dict[UOp, int] = {}
        layout = []
        for j in range(unroll):
            idx = []
            for w in fetch(*place_of(ph, j), part):
                idx.append(unique.setdefault(w.cast(u32), len(unique)))
            layout.append(idx)
        return list(unique), layout

    def held(values: list[UOp], dtype: Any) -> UOp:
        reg = UOp.alloc((len(values),), dtype, addrspace=AddrSpace.REG)
        return reg.after(reg.store(UOp.stack(*(v.cast(dtype) for v in values))))

    first_words, layout = block_words(UOp.const(0, dtypes.weakint))
    held_w = held(first_words, u32) if first_words and n_steps > unroll else None  # one block: loaded in the loop

    a_lds = UOp.alloc((tile, 32 + WIDE_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
    chunk_ids = [tid + r * threads for r in range(-(-chunks // threads))]
    places = [((chunk // per_row).minimum(tile - 1), (chunk % per_row) * 8) for chunk in chunk_ids]
    keep = [(row < count) & (chunk < chunks) for chunk, (row, _) in zip(chunk_ids, places, strict=True)]
    lds_places = [a_lds[row, c] for row, c in places]

    def fetch_a(i: int, s: UOp | int, offset: UOp | int = 0) -> UOp:
        row, c = places[i]
        if WIDE_DIAG in ("inputs", "both"):
            return UOp.stack(*((c + e).cast(dtypes.half) for e in range(8)))
        return _amd_load(x[pair(row) // x_share + offset, s * 32 + c], 8)  # rows past `count` read its last row

    # WIDE_ACT_DEPTH sets of a step's activations in flight: a load is waited for that many steps after it was
    # issued, and the in-order load counter then waits for the weight loads issued before it too
    depth = min(WIDE_ACT_DEPTH, n_steps)
    held_a = [
        [held([fetch_a(i, d)[e] for e in range(8)], dtypes.half) for i in range(len(places))] for d in range(depth)
    ]
    token = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    token = token.after(token.store(UOp.const(0, dtypes.int32)))

    phase = UOp.range(live.where(n_phases, 0), 4)
    next_words, next_layout = block_words((phase + 1).minimum(n_phases - 1))  # the last block loads itself again
    if next_layout != layout or len(next_words) != len(first_words):
        raise ValueError("a block's words do not have the same layout in every phase")
    if held_w is not None:
        words = [held_w.after(phase)[i].load() for i in range(len(first_words))]
    else:  # the only block's words (or none, for the codebook types, which load in `convert`)
        words = block_words(phase)[0] if first_words else []
    pending = [[UOp.stack(*(r.after(phase)[e].load() for e in range(8))) for r in regs] for regs in held_a]
    tok = token.after(phase)[0].load()

    accs = [UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(tiles16)]
    accs = [a.after(a.store(a.const_like(0))) for a in accs]
    values = [UOp.stack(*(a.after(phase)[i].load() for i in range(8))) for a in accs]
    reads: list[UOp] = []
    first_store = None
    for j in range(unroll):
        s = phase * unroll + j
        stored = _lds_store(tok, lds_places, pending.pop(0), keep)
        first_store = first_store if first_store is not None else stored
        ready = _token(_LDS_FENCE, stored)
        zero = _opaque_zero(ready)
        pending.append([fetch_a(i, (s + depth).minimum(n_steps - 1), zero) for i in range(len(places))])
        if WIDE_DIAG in ("weights", "both"):
            halves = [[(s * 2 + k16 + i).cast(u32) | 0x3C003C00 for i in range(4)] for k16 in range(2)]
        else:
            base, group = place_of(phase, j)
            halves = convert([words[i] for i in layout[j]], base, group, part)
        afrags = [[UOp.stack(*own, *(_swap_halves(w) for w in own)).bitcast(dtypes.half)] for own in halves]
        reads += [a for frags in afrags for a in frags]
        done = []
        for tt in range(tiles16):  # every tile has a first row
            run = None if tt == 0 else count > tt * 16
            got, block = _tile_step(afrags, a_lds[tt * 16 + col, 0], [values[tt]], run, ready)
            done.append(block)
            values[tt] = got[0]
        tok = _token(_LDS_FENCE, *done)  # every wave's reads of this step, before the next step's writes
    updates = [a.store(values[tt]) for tt, a in enumerate(accs)]
    # after step 0 used the sets loaded last phase
    updates += [
        r.after(first_store).store(a)
        for regs, acts in zip(held_a, pending, strict=True)
        for r, a in zip(regs, acts, strict=True)
    ]
    if held_w is not None:
        updates.append(held_w.after(*reads).store(UOp.stack(*next_words)))
    updates.append(token.store(tok))
    update = UOp.group(*updates).end(phase)

    low = lane_group.eq(0)
    out_stores = []
    for tt in range(tiles16):
        r = tt * 16 + col
        dest = pair(r).valid(live & (r < count))
        h = [accs[tt].after(update)[v].load() for v in range(8)]
        # low lanes keep outputs 0, 2, 4, 6 and take 1, 3, 5, 7; high lanes keep 9, .., 15 and take 8, .., 14
        sent = [low.where(h[4 + i], h[i]) for i in range(4)]
        took = [_swap_halves(v.bitcast(u32)).bitcast(dtypes.float32) for v in sent]
        ordered = [v for i in range(4) for v in (low.where(h[i], took[i]), low.where(took[i], h[4 + i]))]
        first = output_block * BN + wave * 16 + lane_group * 8
        if gate_in is not None:
            gates = [gate_in[pair(r), first + j_].load() for j_ in range(8)]
            ordered = [(g * g.sigmoid() * u).cast(dtypes.half) for g, u in zip(gates, ordered, strict=True)]
        elif half:
            ordered = [v.cast(dtypes.half) for v in ordered]
        out_stores += [out[dest, first + j_].store(v) for j_, v in enumerate(ordered)]
    flops = 2 * slots * tile * out_features * in_features
    weight_bytes = experts * expert_words * raw.dtype.itemsize
    estimates = Estimates(ops=flops, lds=weight_bytes, mem=weight_bytes + slots * tile * out_features * 4)
    name = f"moe_block_{NAMES[ggml_type]}{'_swiglu' if swiglu else ''}_t{tile}"
    sink = UOp.group(*out_stores).end(wave, lane).end(tile_slot, output_block)
    return sink.sink(arg=KernelInfo(name=name, opts_to_apply=(), estimates=estimates, optimized=True))


def _prefetch(ggml_type: int, raw: UOp, raw_up: UOp | None, offset: UOp) -> tuple[Any, Any]:
    """`wmma_linear`'s prefetch pair for the tile's expert (`offset` words into the weights), per output tile."""

    def shifted(raw_: UOp) -> tuple[Callable[..., list[UOp]], Callable[..., UOp]]:
        fetch, convert = _fetcher(ggml_type, raw_)
        return (lambda base, subgroup, half: fetch(base + offset, subgroup, half)), convert

    if raw_up is None:
        return shifted(raw)
    (fg, cg), (fu, cu) = shifted(raw), shifted(raw_up)
    return (fg,) * OUTPUT_TILES + (fu,) * OUTPUT_TILES, (cg,) * OUTPUT_TILES + (cu,) * OUTPUT_TILES


def _swiglu_stores(
    out: UOp,
    outputs: tuple[Any, ...],
    tokens: tuple[Any, ...],
    accs: tuple[Any, ...],
    update: UOp,
    lane: UOp,
    wave: UOp,
    waves: int,
    kgroups: int,
) -> list[UOp]:
    """Stores of SwiGLU(gate, up) as fp16: accs holds the gate's output tiles, then the up matrix's, in the same
    fragment layout, so they combine element by element before tinygrad's RDNA3 exchange of fragment halves."""
    if kgroups != 1:
        raise ValueError("the fused gate/up stores follow the RDNA3 fragment layout (kgroups 1)")
    n, tt = len(accs) // 2, len(tokens)
    lds = UOp.alloc((waves, 32, n * tt * 8), dtypes.float32, addrspace=AddrSpace.LOCAL)
    stores = []
    for o in range(n):
        for t in range(tt):
            for i in range(8):
                g, u = accs[o][t].after(update)[i].load(), accs[n + o][t].after(update)[i].load()
                stores.append(lds[wave, lane, (o * tt + t) * 8 + i].store(g * g.sigmoid() * u))
    lds = lds.after(*stores)
    low = (lane // 16).eq(0)

    def values(ai: int) -> tuple[UOp, ...]:
        own = tuple(lds[wave, lane, ai * 8 + i].load() for i in range(8))
        peer = tuple(lds[wave, lane ^ 16, ai * 8 + i].load() for i in range(8))
        return tuple(
            low.where(own[i], peer[i + 4]) if j == 0 else low.where(peer[i], own[i + 4])
            for i in range(4)
            for j in range(2)
        )

    return [
        out[token, outputs[o]].store(value.cast(dtypes.half))
        for o in range(n)
        for t, tile_tokens in enumerate(tokens)
        for token, value in zip(tile_tokens, values(o * tt + t), strict=True)
    ]


def find(layer: Any) -> tuple[int, Tensor] | None:
    """(ggml type, packed words) of an expert layer this module serves, read from its decoded weights."""
    if layer.weight_scale_inv is not None or hasattr(layer, "bias") or layer.weight.uop.axis is not None:
        return None
    dev = layer.weight.device
    if layer.in_features % 32 or not amd_custom_kernels_supported(dev) or _is_rdna4(dev) or _is_cdna(dev):
        return None
    found = find_packed(layer.weight, layer.in_features) or moe_small.find_small(layer.weight, layer.in_features)
    if found is None or found[0] not in TYPE_WORDS or (found[0] in BLOCKS_256 and layer.in_features % 256):
        return None
    return found


def _output_block() -> int:
    """Outputs per workgroup (both kernels' blocks)."""
    return math.lcm(16 * OUTPUT_TILES * WAVES, 16 * WIDE_WAVES if WIDE_TILES else 1)


def serves(layer: Any, sel: Tensor) -> bool:
    """Whether this call runs here: enough tokens, and a layer `find` recognised."""
    tokens = int(sel.max_shape[0]) * int(sel.max_shape[1])
    out_features = int(layer.weight.shape[1])
    return tokens >= MIN_TOKENS and out_features % _output_block() == 0


def route(sel: Tensor, experts: int, tokens: int) -> tuple[list[tuple[int, tuple[Tensor, ...]]], Tensor]:
    """([(tile rows, (expert, first row, rows, total))] per class, rows) for the flat pair list `sel` (int32, -1
    for padding)."""
    (P,) = sel.shape
    if P % ROUTE_VECTOR:  # padding pairs match no expert
        sel = sel.pad(((0, -P % ROUTE_VECTOR),), value=-1).contiguous()
    if WIDE_TILES:  # every expert with more than SMALL_MAX rows in wide tiles, the others in one small tile
        big, small = WIDE, SMALL_MAX // SMALL
        # whole wide tiles hold WIDE rows, and at most one other per expert with more than SMALL_MAX rows
        first_slots = P // WIDE + min(experts, P // (SMALL_MAX + 1))
    else:  # 64-row tiles hold more than 48 rows but one per expert
        big, small = BIG, BIG // SMALL - 1
        first_slots = P // (BIG - SMALL) + min(experts, P)
    sizes = [big] + [SMALL * c for c in range(small, 0, -1)]
    slots = [max(1, first_slots)] + [max(1, min(experts, P))] * small  # the small tiles: one per expert
    i32 = functools.partial(Tensor.empty, dtype=dtypes.int32, device=sel.device)
    outs = [i32(P)]
    for n in slots:
        outs += [i32(n), i32(n), i32(n), i32(1)]
    most_full = max(1, -(-tokens // big))
    fxn = functools.partial(_route_kernel, experts=experts, most_full=most_full, big=big, small=small, wide=WIDE_TILES)
    rows, *tables = Tensor.custom_kernel(*outs, sel, fxn=fxn)[: len(outs)]
    return [(size, tuple(tables[4 * i : 4 * i + 4])) for i, size in enumerate(sizes)], rows


Routing = tuple[list[tuple[int, tuple[Tensor, ...]]], Tensor]


def routing_of(sel: Tensor, experts: int) -> Routing:
    """`route` for sel (B, T, k), on the padded maximum when the token count is symbolic (as the grouped calls run)."""
    if not isinstance(sel.shape[1], int):
        sel = sel.pad_to(sel.max_shape, value=-1)
    return route(sel.cast(dtypes.int32).flatten().contiguous(), experts, int(sel.shape[0]) * int(sel.shape[1]))


def grouped_gate_up(gate: Any, up: Any, sel: Tensor, x: Tensor, routing: Routing | None = None) -> Tensor:
    """SwiGLU(gate(x), up(x)) of the selected experts in fp16: sel (B, T, k), x (B, T, 1, in) -> (B, T, k, out).
    `routing` is `routing_of(sel, experts)` when the caller shares it with the down projection."""
    if not isinstance(T := sel.shape[1], int):  # symbolic token count: run on the padded maximum
        padded = grouped_gate_up(gate, up, sel.pad_to(sel.max_shape, value=-1), x.pad_to(x.max_shape), routing)
        return padded[..., :T, :, :]
    (ggml_type, raw), (_, raw_up) = gate._tinyllm_moe_mm, up._tinyllm_moe_mm
    B, T, k = (int(s) for s in sel.shape[-3:])
    in_features = int(x.shape[-1])
    experts, out_features = int(gate.weight.shape[0]), int(gate.weight.shape[1])
    classes, rows = routing if routing is not None else routing_of(sel, experts)
    grid = (_iq_grid(x.device, ggml_type),) if ggml_type in GRID_TYPES else ()
    xh = x.reshape(B * T, in_features).cast(dtypes.half).contiguous()
    out = Tensor.empty(B * T * k, out_features, dtype=dtypes.half, device=x.device)
    common = dict(out_features=out_features, in_features=in_features, ggml_type=ggml_type, x_share=k)
    for i, (tile, (expert, start, count, total)) in enumerate(classes):
        if (plan := _class_plan(i, tile, B * T)) is None:
            continue
        kind, height = plan
        tables = (rows, expert, start, total, count)
        if kind == "block":  # the gate into a scratch buffer, then the up projection with SwiGLU in its stores
            gate_out = Tensor.empty(B * T * k, out_features, dtype=dtypes.float32, device=x.device)
            fxn = functools.partial(_block_kernel, tile=height, **common)
            gate_out = Tensor.custom_kernel(gate_out, raw, xh, *tables, *grid, fxn=fxn)[0]
            fxn = functools.partial(_block_kernel, tile=height, swiglu=True, **common)
            out = Tensor.custom_kernel(out, raw_up, xh, *tables, gate_out, *grid, fxn=fxn)[0]
            continue
        fxn = functools.partial(_grouped_kernel, tile=height, fused=True, **common)
        out = Tensor.custom_kernel(out, raw, xh, *tables, raw_up, *grid, fxn=fxn)[0]
    return out.reshape(B, T, k, out_features)


def _class_plan(i: int, tile: int, tokens: int) -> tuple[str, int] | None:
    """(kernel, rows per tile) for tile class i (tiles of up to `tile` rows) in a call of `tokens` tokens, or None
    when no expert fills it: "block" (`_block_kernel`) or "grouped" (`_grouped_kernel`)."""
    if i == 0 and WIDE_TILES:  # every expert with more than SMALL_MAX rows
        return ("block", WIDE) if tokens > SMALL_MAX else None
    if tile - SMALL >= tokens:  # no expert has that many rows
        return None
    return "grouped", tile


def fused_serves(block: Any, x: Tensor) -> bool:
    """Whether a block's expert FFN runs `moe_forward`: gate and up served here with the same type, a prefill-sized
    call, no expert biases, latent projections or hashed routing."""
    if (
        not (_fused and FUSE_GATE_UP)
        or not hasattr(block, "ffn_gate_exps")
        or int(x.max_shape[0]) * int(x.max_shape[1]) < MIN_TOKENS
    ):
        return False
    cfg, gate, up = block.config, block.ffn_gate_exps, block.ffn_up_exps
    if (
        cfg.expert_latent_dim
        or cfg.expert_proj_bias
        or hasattr(block, "ffn_gate_tid2eid")
        or cfg.expert_group_count > 1
    ):
        return False
    for layer in (gate, up, block.ffn_down_exps):
        if "_tinyllm_moe_mm" not in layer.__dict__:
            layer._tinyllm_moe_mm = find(layer)
    found_g, found_u = gate._tinyllm_moe_mm, up._tinyllm_moe_mm
    same = found_g is not None and found_u is not None and found_g[0] == found_u[0]
    return bool(same and int(gate.weight.shape[1]) % _output_block() == 0)


def moe_forward(block: Any, x: Tensor) -> Tensor:
    """The expert part of `FFNBlock._feed_forward` (tinygrad.llm.model, pinned commit) with gate and up fused."""
    cfg = block.config
    h = x.unsqueeze(2)
    logits = block.ffn_gate_inp(x)
    bias = block.exp_probs_b["bias"] if hasattr(block, "exp_probs_b") else None
    gating, normalize = cfg.expert_gating_func, cfg.norm_topk_prob
    if gating == llm_model.ExpertGating.SOFTMAX and bias is None and normalize:
        gating, normalize = llm_model.ExpertGating.SOFTMAX_WEIGHT, False
    if gating == llm_model.ExpertGating.SOFTMAX_WEIGHT:
        scores = logits
    elif gating == llm_model.ExpertGating.SOFTMAX:
        scores = logits.softmax(-1)
    elif gating == llm_model.ExpertGating.SIGMOID:
        scores = logits.sigmoid()
    else:
        scores = logits.softplus().sqrt()
    route_scores = scores if bias is None else scores + bias
    _, sel = llm_model.pairwise_topk(route_scores, cfg.num_experts_per_tok)
    probs = scores.gather(-1, sel)
    if gating == llm_model.ExpertGating.SOFTMAX_WEIGHT:
        probs = probs.softmax(-1)
    if normalize:
        probs = probs / probs.sum(axis=-1, keepdim=True)
    probs = probs * cfg.routed_scaling_factor
    gate, down = block.ffn_gate_exps, block.ffn_down_exps
    routing = routing_of(sel, int(gate.weight.shape[0]))  # gate/up and down group the same pairs
    act = grouped_gate_up(gate, block.ffn_up_exps, sel, h, routing)
    found = down.__dict__.get("_tinyllm_moe_mm")
    if found is not None and serves(down, sel):
        x_down = grouped_linear(down, found, sel, act, routing, half=HALF_DOWN)  # (B, T, k, D)
    else:
        x_down = down(sel, act)
    out = (x_down * probs.unsqueeze(-1)).sum(axis=-2)
    if hasattr(block, "ffn_gate_shexp"):
        shared = llm_model.gated_activation(block.ffn_gate_shexp(x), block.ffn_up_shexp(x), cfg).contiguous()
        shexp = block.ffn_down_shexp(shared)
        if hasattr(block, "ffn_gate_inp_shexp"):
            shexp = shexp * (x * block.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
    return out


_fused = False


def install() -> None:
    """Run the expert FFN of served blocks through `moe_forward` (once per process)."""
    global _fused
    _fused = True
    block_cls = llm_model.FFNBlock
    if getattr(block_cls._feed_forward, "_tinyllm_moe_mm", False):
        return
    original = block_cls._feed_forward

    def _feed_forward(self: Any, x: Tensor, tokens: Tensor | None = None) -> Tensor:
        return moe_forward(self, x) if fused_serves(self, x) else original(self, x, tokens)

    _feed_forward._tinyllm_moe_mm = True  # type: ignore[attr-defined]
    block_cls._feed_forward = _feed_forward


def grouped_linear(
    layer: Any,
    found: tuple[int, Tensor],
    sel: Tensor,
    x: Tensor,
    routing: Routing | None = None,
    half: bool = False,
) -> Tensor:
    """`ExpertWeights.__call__` for prefill: sel (B, T, k), x (B, T, 1 or k, in) -> (B, T, k, out), in fp32, or in
    fp16 with `half` when every tile runs `_block_kernel`."""
    if not isinstance(T := sel.shape[1], int):  # symbolic token count: run on the padded maximum
        padded = grouped_linear(layer, found, sel.pad_to(sel.max_shape, value=-1), x.pad_to(x.max_shape), routing, half)
        return padded[..., :T, :, :]
    ggml_type, raw = found
    B, T, k = (int(s) for s in sel.shape[-3:])
    in_features, x_slots = int(x.shape[-1]), int(x.shape[-2])
    experts, out_features = int(layer.weight.shape[0]), int(layer.weight.shape[1])
    classes, rows = routing if routing is not None else routing_of(sel, experts)
    grid = (_iq_grid(x.device, ggml_type),) if ggml_type in GRID_TYPES else ()
    xh = x.reshape(B * T * x_slots, in_features).cast(dtypes.half).contiguous()
    half = half and WIDE_TILES and SMALL_MAX == 0  # the per-wave kernel stores fp32
    out = Tensor.empty(B * T * k, out_features, dtype=dtypes.half if half else dtypes.float32, device=x.device)
    common = dict(out_features=out_features, in_features=in_features, ggml_type=ggml_type, x_share=k // x_slots)
    for i, (tile, (expert, start, count, total)) in enumerate(classes):
        if (plan := _class_plan(i, tile, B * T)) is None:
            continue
        kind, height = plan
        kernel = _block_kernel if kind == "block" else _grouped_kernel
        fxn = functools.partial(kernel, tile=height, **common, **({"half": True} if half else {}))
        out = Tensor.custom_kernel(out, raw, xh, rows, expert, start, total, count, *grid, fxn=fxn)[0]
    return out.reshape(B, T, k, out_features)
