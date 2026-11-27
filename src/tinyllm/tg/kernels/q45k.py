"""Matrix-core prefill kernel for linear layers stored as Q4_K or Q5_K.

In a 32-token prefill of `Qwen3.8-27B-UD-Q4_K_XL.gguf`, tinygrad's matrix-core kernels for these two types take
49% of the pass (Q5_K 33%, Q4_K 15.5%; `tools/profile_step.py --kind prefill`). They use tinygrad's one-group-
per-step loop. tinygrad's decode kernel for both types is kept. This kernel runs tinygrad's own arithmetic on
tinyllm's loop (wmma.py) with 2 groups of 32 weights per step, so the weights the matrix cores see, and the
outputs, are bit-identical to tinygrad's (`_config`).

Both types store 256 weights per super-block as 32-bit words: word 0 holds the fp16 scales d and dmin, words 1-3
twelve bytes of 6-bit group scales sc and minimums m (one pair per group of 32), then for Q5_K 32 bytes of fifth
bits qh, then 128 bytes of 4-bit codes. A weight is d * sc * q - dmin * m, with q in 0..15 (Q4_K) or 0..31 (Q5_K).

A packed conversion is kept as an option (PACKED = True): v_perm_b32 puts two code bytes under the fp16 exponent
of 1024.0, giving 1024 + q exactly, and one v_pk_fma_f16 computes (h - 1024) * ds + (-dm) for two weights, with
ds = d * sc and dm = dmin * m rounded to fp16 once per group. It is about 3% faster for these layers but rounds
a weight twice where tinygrad rounds once.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.kernels.amd import _is_cdna, _is_rdna4
from tinygrad.llm.kernels.amd_gguf import Q4_K, Q5_K, QUANT_SIZES, _half, _q5_scales, _wmma_words

from tinyllm.tg.kernels.wmma import (
    bind_rows,
    center_affine,
    magic_pairs,
    splat_half,
    valid_rows,
    wmma_layout,
    wmma_linear,
)

QS_WORD = {Q4_K: 4, Q5_K: 12}  # first word of the 4-bit codes in a super-block
QH_WORD = 4  # Q5_K: first word of the fifth bits
NAMES = {Q4_K: "q4_k", Q5_K: "q5_k"}
# Overrides for experiments (tools/prefill_kernel_check.py); None selects by row count, see `_config`.
GROUPS_PER_STEP: int | None = None
PACKED: bool | None = None
NO_WEIGHTS = False  # diagnostic: constant weights, no weight loads. Wrong results; shows what the rest costs.
LDS_BYTES: int | None = None  # experiments: local memory reserved per workgroup instead of RESERVE_LDS
RESERVE_LDS = {Q4_K: 0, Q5_K: 0}  # local memory reserved per workgroup (wmma.wmma_stores); none, see `_config`
TILES: tuple[int, int] | None = None  # experiments: (token tile, output tiles per wave) instead of tinygrad's choice
SHARED: tuple[int, int, int] | None = None  # experiments: (token tile, output tiles, waves), activations shared
PREFETCH = False  # experiments: weights loaded one step ahead (wmma_linear(prefetch=...))
SERVES_ROWS = 32  # above this many rows tinygrad's own kernel is as fast or faster (`_config`); None: all rows


def _config(tokens: int, in_features: int, ggml_type: int) -> tuple[int, bool]:
    """(groups per step, packed conversion). Measured inside the model, a 32-token prefill of `UD-Q4_K_XL` with
    fixed-length chunks, ms per pass in each kernel (tools/profile_step.py --set):

                    tinygrad  1 packed  2 unpacked  2 packed  4 packed  8 packed
        Q5_K           53.9      52.0        43.8      42.3      47.9      45.9
        Q4_K           25.1      24.9        19.2         -      18.8      19.9

    Two groups per step with tinygrad's arithmetic keep the results exact for 1.5 ms (Q5_K) and 0.4 ms (Q4_K)
    more than the best packed form. Per layer in isolation (tools/prefill_kernel_check.py) the packed forms
    looked better, up to 4 groups at 128 rows; in the model those turn into losses: a 128-token prefill spent
    134.6 ms in the Q5_K kernel with 4 groups per step against 122.9 in tinygrad's, and 43.6 against 41.3 for
    Q4_K. The check tool reruns one layer, whose weights partly stay in the 32 MB last-level cache, while in the
    model every layer's weights come from memory. Above 32 rows the matrix-core work itself dominates as well
    (with constant weights a 128-row deltanet qkv call keeps 0.44 of its 0.64 ms), so `q45k_linear` leaves those
    rows to tinygrad's kernel (`SERVES_ROWS`).

    Local memory reserved per workgroup to limit occupancy (`RESERVE_LDS`; densemm.py explains why that helps),
    ms per 32-token pass in each kernel inside the model:

        reserved     none   16 KB   24 KB   32 KB
        Q5_K         43.5    46.7    49.4    47.4
        Q4_K         19.1    17.4    17.0    16.3

    with the same reservation for IQ4_XS (iq4xs.py) in each column. Reserving only for Q4_K did not hold up: with
    32 KB for Q4_K and 16 KB for IQ4_XS the Q4_K kernel took 22.3 ms and `tinyllm bench` gave 256.4 tok/s against
    261.9 without; with 16 KB for both, 20.4 and 20.8 against 19.1 (twice each). The whole pass stayed within 1 ms
    either way, so neither type reserves anything. At chunk 128, Q5_K stayed slower than tinygrad's kernel with
    every reservation (150.5 ms or more against 123.9) and Q4_K was faster only with 16 KB (37.9 against 43.8),
    under 2% of the pass; all Q4_K and Q5_K rows above 32 stay with tinygrad.
    """
    groups = GROUPS_PER_STEP or 2
    packed = False if PACKED is None else PACKED
    return (groups if (in_features // 32) % groups == 0 else 1), packed


@functools.cache
def _q45k_wmma_kernel(
    out: UOp,
    raw: UOp,
    x: UOp,
    *,
    out_features: int,
    in_features: int,
    ggml_type: int,
    kgroups: int,
    lanes: int,
    groups_per_step: int,
    packed: bool,
    no_weights: bool = False,
    lds_bytes: int = 0,
    tiles: tuple[int, int] | None = None,
    shared: tuple[int, int, int] | None = None,
    prefetch: bool = False,
    rows: Any = None,
) -> UOp:
    tokens = int(out.shape[0])
    # tinygrad's tiling for these types (_q5_linear_f16_wmma_kernel)
    waves = None
    if shared is not None and (tokens % shared[0] or out_features % (16 * shared[1] * shared[2])):
        shared = None  # does not fit this layer: the usual loop
    if shared is not None:
        token_tile, output_tiles, waves = shared
    elif tiles is not None:
        token_tile, output_tiles = tiles
    elif tokens % 64 == 0:
        token_tile, output_tiles = 64, 1 if out_features <= 1024 else 2
    else:
        token_tile, output_tiles = 32 if tokens % 32 == 0 else 16, 2
    layout = wmma_layout(out, out_features, token_tile, output_tiles, lanes, waves)
    word_indices = _wmma_words(layout[5], kgroups)
    q5 = ggml_type == Q5_K

    def dequant(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
        # `base` addresses the super-block, `subgroup` (0..7) its 32 weights, `half` 16 of them
        if no_weights:
            return tuple(UOp.const(0.001 * (k + 1), dtypes.half) for k in range(4 * len(word_indices)))
        d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
        qs = base + QS_WORD[ggml_type] + (subgroup // 2) * 8 + half * 4
        shift = ((subgroup & 1) * 4).cast(dtypes.uint32)
        codes = []
        for i in word_indices:
            code = (raw[qs + i] >> shift) & 0x0F0F0F0F
            if q5:
                code = code | (((raw[base + QH_WORD + half * 4 + i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4)
            codes.append(code)
        if packed:
            ds2 = splat_half((d * scale).cast(dtypes.half))
            dm2 = splat_half((-(dmin * minimum)).cast(dtypes.half))
            words = [center_affine(pair, ds2, dm2, 1024) for code in codes for pair in magic_pairs(code)]
            return UOp.stack(*words).bitcast(dtypes.half)  # (n,) words -> (2n,) halves
        return tuple(
            (((code >> (8 * b)) & 255).float() * d * scale - dmin * minimum).cast(dtypes.half)
            for code in codes
            for b in range(4)
        )

    def fetch(base: UOp, subgroup: UOp, half: int) -> list[UOp]:
        # d and dmin, the three scale words, this half's code words, and for Q5_K its fifth-bit words
        qs = base + QS_WORD[ggml_type] + (subgroup // 2) * 8 + half * 4
        words = [raw[base + i] for i in range(4)] + [raw[qs + i] for i in word_indices]
        return words + ([raw[base + QH_WORD + half * 4 + i] for i in word_indices] if q5 else [])

    def convert(words: list[UOp], subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
        head, w1, w2, w3 = words[:4]
        sb = ((subgroup & 3) * 8).cast(dtypes.uint32)
        byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
        scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4)).float()
        minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4)).float()
        d, dmin = _half(head & 0xFFFF), _half(head >> 16)
        shift = ((subgroup & 1) * 4).cast(dtypes.uint32)
        n = len(word_indices)
        codes = []
        for i in range(n):
            code = (words[4 + i] >> shift) & 0x0F0F0F0F
            if q5:
                code = code | (((words[4 + n + i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4)
            codes.append(code)
        if packed:
            ds2 = splat_half((d * scale).cast(dtypes.half))
            dm2 = splat_half((-(dmin * minimum)).cast(dtypes.half))
            out_words = [center_affine(pair, ds2, dm2, 1024) for code in codes for pair in magic_pairs(code)]
            return UOp.stack(*out_words).bitcast(dtypes.half)
        return tuple(
            (((code >> (8 * b)) & 255).float() * d * scale - dmin * minimum).cast(dtypes.half)
            for code in codes
            for b in range(4)
        )

    name = (
        f"linear_{NAMES[ggml_type]}_f16_wmma_g{groups_per_step}"
        f"{'_packed' if packed else ''}{'_noweights' if no_weights else ''}"
        + (f"_lds{lds_bytes}" if lds_bytes else "")
        + (f"_shared{shared[0]}x{shared[1]}x{shared[2]}" if shared else "")
        + ("_prefetch" if prefetch else "")
    )
    type_words = QUANT_SIZES[ggml_type] // 4
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
        share_inputs=shared is not None,
        prefetch=(fetch, convert) if prefetch and not no_weights else None,
        rows=rows,
    )


def find_q45k(layer: Any) -> tuple[int, Tensor] | None:
    """tinygrad's own packed form of the layer's weights if they are Q4_K or Q5_K, else None."""
    from tinygrad.llm.kernels.amd_gguf import find_packed

    found = find_packed(layer.weight, int(layer.in_features))
    return found if found is not None and found[0] in (Q4_K, Q5_K) else None


def q45k_linear(layer: Any, x: Tensor, fallback: Callable[[Any, Tensor], Tensor]) -> Tensor:
    """`layer(x)` for a Q4_K or Q5_K layer: matrix cores for multiples of 16 rows, tinygrad's path otherwise."""
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    tokens = int(prod(x.shape[:-1]))
    overrides = (
        GROUPS_PER_STEP is not None
        or PACKED is not None
        or LDS_BYTES is not None
        or TILES is not None
        or bool(SHARED)
        or PREFETCH
    )  # experiments run on every row count
    above = SERVES_ROWS is not None and tokens > SERVES_ROWS and not overrides
    if tokens % 16 or out_features % 16 or in_features % 256 or above:
        return fallback(layer, x)
    cdna = _is_cdna(x.device)
    kgroups, lanes = (4, 64) if cdna else (2 if _is_rdna4(x.device) else 1, 32)
    groups, packed = _config(tokens, in_features, int(layer.ggml_type))
    rows = valid_rows()
    fxn = functools.partial(
        _q45k_wmma_kernel,
        out_features=out_features,
        in_features=in_features,
        ggml_type=int(layer.ggml_type),
        kgroups=kgroups,
        lanes=lanes,
        groups_per_step=groups,
        packed=packed,
        no_weights=NO_WEIGHTS,
        lds_bytes=RESERVE_LDS[int(layer.ggml_type)] if LDS_BYTES is None else LDS_BYTES,
        tiles=TILES,
        shared=SHARED,
        prefetch=PREFETCH,
        rows=rows,
    )
    out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
    xs = x.cast(dtypes.float16).contiguous()  # exactly tinygrad's form, see q8.py
    result = Tensor.custom_kernel(out, layer.weight, bind_rows(xs, rows), fxn=fxn)[0].reshape(
        *x.shape[:-1], out_features
    )
    return result if layer.bias is None else result + layer.bias
