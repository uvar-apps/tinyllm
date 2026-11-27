"""Matrix-core prefill kernel for linear layers stored as Q6_K.

tinygrad multiplies Q6_K weights in place, and its decode kernel already reads them at the
memory-bandwidth limit; that path is kept. Its prefill kernel, which expands the weights to fp16 for
the matrix cores, made a Q6_K file prefill at 123 tok/s against 181 for Q4_K_M
(docs/01-research-findings.md, section 5.5). Measured on Strix Halo, not assumed:

  - With constant weights the same kernel runs 2 to 6 times faster, so the weight path is the limit.
  - A cheaper conversion alone changed nothing, and the compiler already merges the 2-byte loads of
    the 2-byte-aligned Q6_K blocks into 16-byte loads. tinygrad's loop handles one group of 32 weights
    per step and waits on its loads before the matrix-core work: it was waiting on memory.
  - Handling several groups per step (tinyllm.tg.kernels.wmma) issues their loads together. After
    that, the conversion arithmetic matters: 5 to 6 scalar instructions per weight.

So this kernel combines two changes, chosen per row count from whole-model measurements:

  groups per step  4 or 8 groups of 32 weights per loop step instead of 1;
  conversion       OR-ing a code q (0..63) into the bit pattern of 1024.0 gives the fp16 number
                   1024 + q exactly; subtracting 1056 gives q - 32 exactly; one multiply by the
                   block's scale finishes it. The packed form does this for two weights per
                   instruction (v_perm_b32, v_or_b32, v_pk_add_f16, v_pk_mul_f16).

The scale d * sc is rounded to fp16 once per 16 weights, so a weight is rounded twice (tinygrad rounds
once). The difference is within the fp16 rounding the matrix cores already impose (tools/prefill_kernel_check.py).
The bit layout of Q6_K follows tinygrad's `_quant_word`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.kernels.amd import _is_cdna, _is_rdna4
from tinygrad.llm.kernels.amd_gguf import (
    Q6_K,
    QUANT_SIZES,
    _half,
    _load_byte,
    _load_u32,
    _wmma_layout,
    _wmma_words,
)

from tinyllm.tg.kernels import gemm
from tinyllm.tg.kernels.wmma import FP16_1024, bind_rows, byte_pairs, center_scale, splat_half, valid_rows, wmma_linear

# Overrides for experiments (tools/prefill_kernel_check.py); None selects by row count, see `_config`.
GROUPS_PER_STEP: int | None = None
PACKED: bool | None = None
NO_WEIGHTS = False  # diagnostic: constant weights, no weight loads. Wrong results; shows what the rest costs.
GEMM = True  # calls of whole 128-row blocks go to the tiled kernel (gemm.py)
LDS_BYTES: int | None = None  # experiments: local memory reserved per workgroup instead of RESERVE_LDS
TILES: tuple[int, int] | None = None  # experiments: (token tile, output tiles per wave) instead of the default
# Local memory reserved per workgroup (wmma.wmma_stores). In a `UD-Q4_K_XL` pass, ms in this kernel at chunk
# 32 / 128: none 12.2 / 26.2, 16 KB 14.0 / 25.4, 24 KB 13.2 / 30.9, 32 KB 13.4 / 30.8; so none.
RESERVE_LDS = 0


def _config(tokens: int, in_features: int) -> tuple[int, bool]:
    """(groups per step, packed conversion). Whole-model Q6_K prefill on Strix Halo, tok/s:

    chunk 32:  tinygrad 123.3, 8 groups 151.9, 8 groups packed 161.5, 4 groups 150.5
    chunk 128: tinygrad 209.2, 8 groups 248.0, 8 groups packed 242.8, 4 groups 252.2
    """
    groups, packed = (8, True) if tokens <= 32 else (4, False)
    groups = GROUPS_PER_STEP or groups
    packed = packed if PACKED is None else PACKED
    return (groups if (in_features // 32) % groups == 0 else 1), packed


def _codes(raw: UOp, base: UOp, subgroup: UOp, word: int) -> UOp:
    """Four 6-bit codes, one per byte, for weights 4*word .. 4*word+3 of the 32 at `subgroup` (tinygrad's layout)."""
    low = _load_u32(raw, base, (subgroup // 4) * 64 + (subgroup % 2) * 32 + word * 4, stream=True)
    high = _load_u32(raw, base, 128 + (subgroup // 4) * 32 + word * 4, stream=True)
    low = low >> ((subgroup % 4 // 2) * 4).cast(dtypes.uint32)
    high = high >> ((subgroup % 4) * 2).cast(dtypes.uint32)
    return (low & 0x0F0F0F0F) | ((high & 0x03030303) << 4)


@functools.cache
def _q6k_wmma_kernel(
    out: UOp,
    raw: UOp,
    x: UOp,
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
    tokens = int(out.shape[0])
    token_tile, output_tiles = 64 if tokens % 64 == 0 else 32 if tokens % 32 == 0 else 16, 2
    if tiles is not None and tokens % tiles[0] == 0:
        token_tile, output_tiles = tiles
    layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes)
    word_indices = _wmma_words(layout[5], kgroups)
    offset = UOp.const(1024 + 32, dtypes.half)

    def dequant(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
        # Q6_K block, 256 weights in 210 bytes: ql[128] low nibbles, qh[64] high bit pairs, sc[16] int8, d fp16.
        # `subgroup` (0..7) picks 32 weights, `half` 16 of them.
        if no_weights:
            return tuple(UOp.const(0.001 * (k + 1), dtypes.half) for k in range(4 * len(word_indices)))
        sc = _load_byte(raw, base, 192 + subgroup * 2 + half).cast(dtypes.uint8).bitcast(dtypes.int8).float()
        scale = (_half(raw[base + 104]) * sc).cast(dtypes.half)
        if packed:
            scale2 = splat_half(scale)
            words = [
                center_scale(pair | (FP16_1024 | FP16_1024 << 16), scale2, 1024 + 32)
                for i in word_indices
                for pair in byte_pairs(_codes(raw, base, subgroup, half * 4 + i))
            ]
            return UOp.stack(*words).bitcast(dtypes.half)  # (n,) words -> (2n,) halves
        values = []
        for i in word_indices:
            codes = _codes(raw, base, subgroup, half * 4 + i)
            for b in range(4):
                h = (((codes >> (8 * b)) & 0x3F) | FP16_1024).cast(dtypes.uint16).bitcast(dtypes.half)  # 1024 + q
                values.append((h - offset) * scale)  # (q - 32), exact, times the scale
        return tuple(values)

    name = (
        f"linear_q6_k_f16_wmma_g{groups_per_step}{'_packed' if packed else ''}{'_noweights' if no_weights else ''}"
        + (f"_lds{lds_bytes}" if lds_bytes else "")
        + (f"_t{tiles[0]}x{tiles[1]}" if tiles else "")
    )
    type_words = QUANT_SIZES[Q6_K] // 2
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
        lds_bytes=lds_bytes,
        rows=rows,
    )


def find_q6k(layer: Any) -> tuple[int, Tensor] | None:
    """tinygrad's own packed form of the layer's weights if they are Q6_K, else None."""
    from tinygrad.llm.kernels.amd_gguf import find_packed

    found = find_packed(layer.weight, int(layer.in_features))
    return found if found is not None and found[0] == Q6_K else None


def q6k_linear(layer: Any, x: Tensor, fallback: Callable[[Any, Tensor], Tensor]) -> Tensor:
    """`layer(x)` for a Q6_K layer: matrix cores for multiples of 16 rows, tinygrad's path (`fallback`) otherwise."""
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    tokens = int(prod(x.shape[:-1]))
    if tokens % 16 or out_features % 16:
        return fallback(layer, x)
    if GEMM and gemm.serves(layer, x, Q6_K):  # large prefill calls: the tiled kernel
        return gemm.gemm_linear(layer, layer.weight, Q6_K, x)
    cdna = _is_cdna(x.device)
    kgroups, lanes = (4, 64) if cdna else (2 if _is_rdna4(x.device) else 1, 32)
    groups, packed = _config(tokens, in_features)
    rows = valid_rows()
    fxn = functools.partial(
        _q6k_wmma_kernel,
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
    # exactly tinygrad's form (no reshape first): layers sharing an input then share one fp16 copy of it, so
    # mixing these kernels with tinygrad's does not change how the producer of that input is scheduled
    xs = x.cast(dtypes.float16).contiguous()
    result = Tensor.custom_kernel(out, layer.weight, bind_rows(xs, rows), fxn=fxn)[0].reshape(
        *x.shape[:-1], out_features
    )
    return result if layer.bias is None else result + layer.bias
