"""Matrix-core prefill kernel for linear layers stored as Q8_0.

As for Q6_K (q6k.py), tinygrad's decode kernel already reads Q8_0 weights at the memory-bandwidth limit
and is kept, while its prefill kernel takes 85% of a 32-token pass of the Q8_0 Qwen3.8-27B file
(`tools/profile_step.py --kind prefill`). Q8_0 runs through the same tinygrad matrix-core loop, which
waits on one group's loads per step, and its 34-byte blocks are also only 2-byte aligned. This kernel
uses tinyllm's loop (wmma.py) with a packed fp16 conversion; for Q8_0 the packing matters most (`_config`).

Q8_0 block: an fp16 scale d, then 32 int8 codes q; a weight is d * q. A code byte XOR 0x80 is the
unsigned number q + 128, so OR-ing it into the bit pattern of 1024.0 gives the fp16 number 1152 + q
exactly, and subtracting 1152 gives q exactly. d is already fp16, so d * q is rounded once, exactly as
tinygrad's own path rounds it: the weights fed to the matrix cores are bit-identical.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.kernels.amd import _is_cdna, _is_rdna4
from tinygrad.llm.kernels.amd_gguf import Q8_0, QUANT_SIZES, _load_u32, _wmma_words

from tinyllm.tg.kernels.wmma import FP16_1024, byte_pairs, center_scale, splat_half, wmma_layout, wmma_linear

BLOCK_HALFWORDS = 17  # 34 bytes per 32 weights
OFFSET = 1024 + 128
# Overrides for experiments (tools/prefill_kernel_check.py); None selects by row count, see `_config`.
GROUPS_PER_STEP: int | None = None
PACKED: bool | None = None
NO_WEIGHTS = False  # diagnostic: constant weights, no weight loads. Wrong results; shows what the rest costs.
LDS_BYTES: int | None = None  # experiments: local memory reserved per workgroup instead of RESERVE_LDS
TILES: tuple[int, int] | None = None  # experiments: (token tile, output tiles per wave) instead of the default
SHARED: tuple[int, int, int] | None = None  # experiments: (token tile, output tiles, waves), activations shared
PREFETCH = False  # experiments: weights loaded one step ahead (wmma_linear(prefetch=...))
# The compiler's own interleaving strategy for matrix loops, above 32 rows (`_config`); None to leave it out.
HINT_ABOVE_32: str | None = "__builtin_amdgcn_iglp_opt(1);"
RESERVE_LDS = 24576  # local memory reserved per workgroup to limit occupancy (wmma.wmma_stores); `_config`


def _config(tokens: int, in_features: int) -> tuple[int, bool]:
    """(groups per step, packed conversion), from per-layer measurements on Qwen3.8-27B shapes (ms per call):

                   32 rows: tinygrad  1 group packed  2 groups packed    128 rows: tinygrad  1 group packed
    ffn gate/up             0.859     0.659           0.587                        1.710     1.390
    ffn down                0.640     0.580           0.537                        1.252     1.311
    attn q                  0.608     0.448           0.427                        1.268     1.058
    deltanet qkv            0.477     0.355           0.358                        1.021     0.870

    Unlike Q6_K, more groups per step lose above 32 rows (4 groups unpacked: up to 2 times slower at 128 rows),
    so for Q8_0 the conversion arithmetic, not memory latency, is the larger cost.

    Local memory reserved per workgroup to limit occupancy (`RESERVE_LDS`; densemm.py explains why that helps),
    ms per prefill pass in this kernel inside the model, `UD-Q8_K_XL` (tools/profile_step.py --set):

        reserved     none   16 KB   24 KB   28 KB   32 KB   40 KB   64 KB
        chunk 32    136.4   130.9   125.6   127.4   126.7   143.1   141.8
        chunk 128   313.9   312.9   297.3   295.7   299.6   384.4   381.9

    With 24 KB, 4 groups per step at chunk 32 (178.1) and 2 at chunk 128 (442.8) still lose. The weights the
    matrix cores see are unchanged, so the outputs stay bit-identical.

    Tiles at chunk 128 (`TILES`, 24 KB): 128 tokens and one output tile per wave took 285.4, 293.3 and 290.5 ms
    against 300.0 to 300.5, but the pass only shortened by about 2 ms (0.6%), as the kernels around it slowed, and
    together with the same tile for densemm the kernel varied from 277.7 to 304.7 between identical runs. 128 x 2
    (378.5) and 64 x 1 (359.5) lose. With 128 x 1, other reservations lose (none 343.6, 16 KB 353.3, 40 KB 331.3).
    tinygrad's tiles stay.

    At 128 rows, per layer: activations shared in local memory, weights loaded a step ahead, and both, are 6 to 30%
    slower than this loop (ffn gate/up 1.26 ms against 1.37 to 1.62). Of the compiler's scheduling hints, every
    explicit interleaving of loads and matrix instructions was slower (up to 1.9 ms), and its own strategy for
    matrix loops (`__builtin_amdgcn_iglp_opt(1)`, `HINT_ABOVE_32`) saved 1.5% in the model: 297.5 to 292.3 ms per
    128-token `UD-Q8_K_XL` pass (twice each), and `tinyllm bench` 343.7 to 348.6 tok/s (`Q8_0` 354.0 to 361.7).
    At 32 rows it changes nothing, and in the Q4_K and Q5_K kernels it was slower, so only this kernel uses it.
    """
    groups, packed = (2, True) if tokens <= 32 else (1, True)
    groups = GROUPS_PER_STEP or groups
    packed = packed if PACKED is None else PACKED
    return (groups if (in_features // 32) % groups == 0 else 1), packed


@functools.cache
def _q8_wmma_kernel(
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
    shared: tuple[int, int, int] | None = None,
    prefetch: bool = False,
    schedule: str | None = None,
) -> UOp:
    tokens = int(out.shape[0])
    token_tile, output_tiles = 64 if tokens % 64 == 0 else 32 if tokens % 32 == 0 else 16, 2
    if tiles is not None and tokens % tiles[0] == 0:
        token_tile, output_tiles = tiles
    if shared is not None and (tokens % shared[0] or out_features % (16 * shared[1] * shared[2])):
        shared = None  # does not fit this layer: the usual loop
    waves = None
    if shared is not None:
        token_tile, output_tiles, waves = shared
    layout = wmma_layout(out, out_features, token_tile, output_tiles, lanes, waves)
    word_indices = _wmma_words(layout[5], kgroups)
    offset = UOp.const(OFFSET, dtypes.half)

    def dequant(base: UOp, subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
        # `base` addresses 8 consecutive blocks (256 weights); `subgroup` picks the block, `half` its 16 weights
        if no_weights:
            return tuple(UOp.const(0.001 * (k + 1), dtypes.half) for k in range(4 * len(word_indices)))
        base = base + subgroup * BLOCK_HALFWORDS
        d = raw[base].load()  # fp16 bits
        words = [_load_u32(raw, base, 2 + (half * 4 + i) * 4, stream=True) for i in word_indices]
        if packed:
            d2 = splat_half(d)
            magic = (FP16_1024 | FP16_1024 << 16) | 0x00800080  # OR the exponent bits, flip the code's sign bit
            out_words = [center_scale(pair ^ magic, d2, OFFSET) for w in words for pair in byte_pairs(w)]
            return UOp.stack(*out_words).bitcast(dtypes.half)  # (n,) words -> (2n,) halves
        scale = d.bitcast(dtypes.half)
        values = []
        for w in words:
            for b in range(4):
                h = ((((w >> (8 * b)) & 0xFF) ^ 0x80) | FP16_1024).cast(dtypes.uint16).bitcast(dtypes.half)  # 1152 + q
                values.append((h - offset) * scale)
        return tuple(values)

    def fetch(base: UOp, subgroup: UOp, half: int) -> list[UOp]:
        base = base + subgroup * BLOCK_HALFWORDS
        d = raw[base].cast(dtypes.uint32)  # fp16 bits
        return [d] + [_load_u32(raw, base, 2 + (half * 4 + i) * 4, stream=True) for i in word_indices]

    def convert(words: list[UOp], subgroup: UOp, half: int) -> tuple[UOp, ...] | UOp:
        d, codes = words[0], words[1:]
        if packed:
            d2 = splat_half(d)
            magic = (FP16_1024 | FP16_1024 << 16) | 0x00800080
            out_words = [center_scale(pair ^ magic, d2, OFFSET) for w in codes for pair in byte_pairs(w)]
            return UOp.stack(*out_words).bitcast(dtypes.half)
        scale = d.cast(dtypes.uint16).bitcast(dtypes.half)
        return tuple(
            (((((w >> (8 * b)) & 0xFF) ^ 0x80) | FP16_1024).cast(dtypes.uint16).bitcast(dtypes.half) - offset) * scale
            for w in codes
            for b in range(4)
        )

    name = (
        f"linear_q8_0_f16_wmma_g{groups_per_step}{'_packed' if packed else ''}{'_noweights' if no_weights else ''}"
        + (f"_lds{lds_bytes}" if lds_bytes else "")
        + (f"_t{tiles[0]}x{tiles[1]}" if tiles else "")
        + (f"_shared{shared[0]}x{shared[1]}x{shared[2]}" if shared else "")
        + ("_prefetch" if prefetch else "")
        + ("_hinted" if schedule else "")
    )
    type_words = QUANT_SIZES[Q8_0] // 2
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
        share_inputs=shared is not None,
        prefetch=(fetch, convert) if prefetch and not no_weights else None,
        schedule=schedule,
    )


def find_q8(layer: Any) -> tuple[int, Tensor] | None:
    """tinygrad's own packed form of the layer's weights if they are Q8_0, else None."""
    from tinygrad.llm.kernels.amd_gguf import find_packed

    found = find_packed(layer.weight, int(layer.in_features))
    return found if found is not None and found[0] == Q8_0 else None


def q8_linear(layer: Any, x: Tensor, fallback: Callable[[Any, Tensor], Tensor]) -> Tensor:
    """`layer(x)` for a Q8_0 layer: matrix cores for multiples of 16 rows, tinygrad's path (`fallback`) otherwise."""
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    tokens = int(prod(x.shape[:-1]))
    if tokens % 16 or out_features % 16:
        return fallback(layer, x)
    cdna = _is_cdna(x.device)
    kgroups, lanes = (4, 64) if cdna else (2 if _is_rdna4(x.device) else 1, 32)
    groups, packed = _config(tokens, in_features)
    fxn = functools.partial(
        _q8_wmma_kernel,
        out_features=out_features,
        in_features=in_features,
        kgroups=kgroups,
        lanes=lanes,
        groups_per_step=groups,
        packed=packed,
        no_weights=NO_WEIGHTS,
        lds_bytes=RESERVE_LDS if LDS_BYTES is None else LDS_BYTES,
        tiles=TILES,
        shared=SHARED,
        prefetch=PREFETCH,
        schedule=HINT_ABOVE_32 if tokens > 32 else None,
    )
    out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
    # exactly tinygrad's form (no reshape first): layers sharing an input then share one fp16 copy of it, so
    # mixing these kernels with tinygrad's does not change how the producer of that input is scheduled
    xs = x.cast(dtypes.float16).contiguous()
    result = Tensor.custom_kernel(out, layer.weight, xs, fxn=fxn)[0].reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
