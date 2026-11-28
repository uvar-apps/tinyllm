"""Matrix-core prefill kernel for linear layers stored as BF16 or F16.

tinyllm's dense decode kernel (dense.py) serves up to 8 rows of activations per read of the weights and
re-reads them for every further 8, so a 32-token prefill reads each BF16 weight four times. This kernel runs
the rows that are a multiple of 16 on the matrix cores, on tinyllm's loop (wmma.py), reading each weight once
per token tile. The weights come straight from the model file's buffer, two per 32-bit word, as in dense.py.

Two ways to feed BF16 weights to the matrix cores (`MODE`):
  bf16  the bf16 matrix instruction (v_wmma_f32_16x16x16_bf16): the weight bits go in unchanged, and the
        activations are rounded to bf16 instead of fp16 (8 significant bits instead of 11). This is the
        default: it is as fast as f16 up to 32 rows and up to 2.2 times faster at 128 (`_config`). llama.cpp's
        CPU path for BF16 weights rounds activations to bf16 as well.
  f16   the fp16 instruction, as for the quantized types: each word is converted to two fp16 weights
        (v_cvt_pkrtz_f16_f32), exactly for every weight of magnitude 2^-14 to 65504.
F16 weights always take the fp16 instruction unchanged. F32 weights (tinygrad's loader rounds them to fp16) are
read as fp32 words and rounded to fp16 the same way, for the layers with many outputs (Flash-Next's router); their
calls of whole 128-row blocks go to the tiled kernel (`GEMM`, gemm.py).

The kernel reserves 64 KB of local memory per workgroup that it does not need (`RESERVE_LDS`), which limits a
workgroup processor to two of its workgroups at a time, one wave per SIMD. That helps in two ways (`_config`):
- The compiler schedules for the occupancy that local memory allows. With only tinygrad's exchange buffer (4 KB
  at 16 rows) that is 16 waves per SIMD, so it keeps register use under 96 and issues each step's loads in three
  rounds, each a full memory latency; with 64 KB it issues all of them at once (118 registers).
- At 32 rows the compiled loop is the same either way, and the kernel is still 1.5 times faster with the
  reservation: fewer workgroups at a time read the weights faster. Why is not established; fewer concurrent
  streams of weight rows is the likely reason.
At 16 to 32 rows the grid has only about 5 waves per SIMD anyway.
"""

from __future__ import annotations

import functools
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.kernels.amd import _amd_load, _is_cdna, _is_rdna4
from tinygrad.llm.kernels.amd_gguf import _wmma_layout, _wmma_words

from tinyllm.tg.kernels.dense import GGML_BF16, GGML_F16, GGML_F32
from tinyllm.tg.kernels.wmma import bind_rows, valid_rows, wmma_linear

ROW_WORDS_PER_BLOCK = 128  # 256 weights of 16 bits
MIN_ROWS = 16  # fewer rows go to the decode kernel
# Smaller layers stay on the decode kernel at every row count: the DeltaNet gate projections (16 to 48 outputs),
# whose results feed exponentials, keep fp32 activations there, and they are too small to gain anything here.
MIN_OUT_FEATURES = 256
# Overrides for experiments (tools/prefill_kernel_check.py); None selects by row count, see `_config`.
GROUPS_PER_STEP: int | None = None
MODE: str | None = None  # "bf16" or "f16", for BF16 weights
NO_WEIGHTS = False  # diagnostic: constant weights, no weight loads. Wrong results; shows what the rest costs.
TILES: tuple[int, int] | None = None  # experiments: (token tile, output tiles per wave)
LDS_BYTES: int | None = None  # experiments: local memory reserved per workgroup instead of RESERVE_LDS
RESERVE_LDS = 65536  # the most a workgroup may use; see the module docstring and `_config`
GEMM = True  # F32 layers: calls of whole 128-row blocks go to the tiled kernel (gemm.py)
_BF16_TO_F16 = (
    "({{ unsigned int tl_w = (unsigned int)({0}); __builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz("
    "__builtin_bit_cast(float, tl_w << 16), __builtin_bit_cast(float, tl_w & 0xFFFF0000u))); }})"
)


def bf16_pair_to_f16(word: UOp) -> UOp:
    """Two BF16 numbers packed in a word as two fp16 numbers, rounding toward zero (exact in fp16's normal range)."""
    return UOp.native(_BF16_TO_F16, shape=(), dtype=dtypes.uint32, fmt=True).call(word, inline=True)


def _config(tokens: int, in_features: int, ggml_type: int) -> tuple[int, str]:
    """(groups of 32 weights per loop step, mode), from measurements on Qwen3.8-27B shapes.

    Per layer (tools/prefill_kernel_check.py --type bf16), ms per call, 2 groups per step, bf16 mode:

                       local memory    16 rows        32 rows         128 rows
                       reserved:       none   64 KB   none   64 KB    none   64 KB
        ffn gate/up                    3.06   0.91    1.67   0.92     2.65   1.77
        ffn down                       1.06   0.94    1.02   0.95     2.70   1.87
        attn q                         1.86   0.68    1.02   0.69     2.06   1.31
        deltanet qkv                   1.48   0.59    0.83   0.57     1.82   1.13

    With 64 KB reserved, 1, 2 and 4 groups per step are within 5% of each other up to 32 rows; 2 is best or within
    3% at 64 and 128 rows, where 4 and 8 lose. Other tiles than tinygrad's gain only on deltanet out (5120 x 6144,
    0.90 against 1.14 ms at 128 rows with one output tile) and lose elsewhere. f16 mode is equal up to 32 rows and
    2.2 times slower at 128 (attn q 2.87 ms), its conversion adding to the vector work the matrix cores share.

    In the model, a prefill pass of `UD-Q8_K_XL` (tools/profile_step.py; its BF16 layers are the attention q, k and
    v projections), ms in those 48 calls: chunk 32, decode kernel 41.3, this kernel 18.1 with 4 KB and 11.2 with
    64 KB; chunk 128, 166.6, 35.1 and 22.7. f16 mode with 64 KB: 11.3 and 34.2.
    """
    groups = GROUPS_PER_STEP or 2
    mode = "f16" if ggml_type in (GGML_F16, GGML_F32) else MODE or "bf16"
    return (groups if (in_features // 32) % groups == 0 else 1), mode


def _tiles(tokens: int, out_features: int) -> tuple[int, int]:
    """(token tile, output tiles per wave): tinygrad's choice for its quantized matrix-core kernel."""
    return (64 if tokens % 64 == 0 else 32 if tokens % 32 == 0 else 16), 2


@functools.cache
def _dense_wmma_kernel(
    out: UOp,
    words: UOp,
    x: UOp,
    *,
    out_features: int,
    in_features: int,
    ggml_type: int,
    mode: str,
    kgroups: int,
    lanes: int,
    groups_per_step: int,
    no_weights: bool = False,
    tiles: tuple[int, int] | None = None,
    lds_bytes: int = 0,
    rows: Any = None,
) -> UOp:
    tokens = int(out.shape[0])
    token_tile, output_tiles = tiles if tiles and tokens % tiles[0] == 0 else _tiles(tokens, out_features)
    layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes)
    word_indices = _wmma_words(layout[5], kgroups)
    weight_dtype = dtypes.bfloat16 if mode == "bf16" else dtypes.half

    def dequant(base: UOp, subgroup: UOp, half: int) -> UOp:
        # a lane's 16 weights (RDNA3) are 8 consecutive words: k = 4 * w + b for w in word_indices, b in 0..3
        if no_weights:
            return UOp.stack(*(UOp.const(0x3C003C00 + k, dtypes.uint32) for k in range(2 * len(word_indices))))
        if ggml_type == GGML_F32:  # one word a weight, rounded to fp16 as tinygrad's loader rounds it
            start = base + subgroup * 32 + half * 16
            loads = [_amd_load(words[start + 4 * i], 4, stream=True) for i in range(4)]
            return UOp.stack(*(p[j].bitcast(dtypes.float32).cast(dtypes.half) for p in loads for j in range(4)))
        start = base + subgroup * 16 + half * 8
        if kgroups == 1:
            parts, width = [_amd_load(words[start + 4 * i], 4, stream=True) for i in range(2)], 4
        else:
            parts, width = [_amd_load(words[start + 2 * w], 2, stream=True) for w in word_indices], 2
        values = [p[j] for p in parts for j in range(width)]
        if ggml_type == GGML_BF16 and mode == "f16":
            values = [bf16_pair_to_f16(v) for v in values]
        return UOp.stack(*values).bitcast(weight_dtype)  # (n,) words -> (2n,) weights, low half first

    kind = "f16" if ggml_type == GGML_F16 else "f32" if ggml_type == GGML_F32 else f"bf16_{mode}"
    suffix = ("_noweights" if no_weights else "") + (f"_lds{lds_bytes}" if lds_bytes else "")
    name = f"linear_{kind}_wmma_g{groups_per_step}{suffix}"
    return wmma_linear(
        out,
        x,
        out_features,
        in_features,
        2 * ROW_WORDS_PER_BLOCK if ggml_type == GGML_F32 else ROW_WORDS_PER_BLOCK,
        layout,
        dequant,
        name,
        kgroups,
        lanes,
        groups_per_step,
        split_blocks=8 % groups_per_step == 0,
        input_dtype=weight_dtype,
        lds_bytes=lds_bytes,
        rows=rows,
    )


def serves(layer: Any, x: Tensor) -> bool:
    """Whether `dense_mm_linear` takes these rows: multiples of 16, at least MIN_ROWS, of a layer with at least
    MIN_OUT_FEATURES outputs."""
    tokens, out_features = int(prod(x.shape[:-1])), int(layer.out_features)
    return tokens >= MIN_ROWS and tokens % 16 == 0 and out_features % 16 == 0 and out_features >= MIN_OUT_FEATURES


def dense_mm_linear(layer: Any, x: Tensor) -> Tensor:
    """`layer(x)` on the matrix cores for a Linear whose weights `dense.find_dense` recognised (`serves` is true)."""
    ggml_type, words = layer._tinyllm_dense
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    tokens = int(prod(x.shape[:-1]))
    cdna = _is_cdna(x.device)
    kgroups, lanes = (4, 64) if cdna else (2 if _is_rdna4(x.device) else 1, 32)
    groups, mode = _config(tokens, in_features, ggml_type)
    rows = valid_rows()
    fxn = functools.partial(
        _dense_wmma_kernel,
        out_features=out_features,
        in_features=in_features,
        ggml_type=ggml_type,
        mode=mode,
        kgroups=kgroups,
        lanes=lanes,
        groups_per_step=groups,
        no_weights=NO_WEIGHTS,
        tiles=TILES,
        lds_bytes=RESERVE_LDS if LDS_BYTES is None else LDS_BYTES,
        rows=rows,
    )
    out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
    xs = x.cast(dtypes.bfloat16 if mode == "bf16" else dtypes.float16).contiguous()
    result = Tensor.custom_kernel(out, words, bind_rows(xs, rows), fxn=fxn)[0].reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
