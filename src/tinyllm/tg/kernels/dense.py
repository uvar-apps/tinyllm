"""Decode kernel for linear layers stored as BF16 or F16.

At the pinned commit tinygrad has no fast path for float weights with more than 2,048 outputs: they
run through generic code generation on a lazily converted copy. For a BF16 GGUF the conversion to
fp16 is redone on every call, so each weight costs 2 bytes read, 2 written and 2 read again, and
Qwen3.8-27B in BF16 decodes at 0.47 tok/s (docs/01-research-findings.md, section 5.5).

This kernel reads the BF16 or F16 bytes straight from the model file's buffer and accumulates in
fp32. Up to 8 rows of activations are served by one read of the weights; more rows are processed 8
at a time, re-reading the weights for each group of 8. A first version that followed tinygrad's
quantized decode layout (one wave per output row, 32 contiguous weights per lane) reached 112 to
168 GB/s for one row and 48 to 65 GB/s for 8: every wave re-read the fp32 activations from cache,
16 bytes of them per byte of weights at 8 rows. This version shares one copy per workgroup.

BF16 to fp32 is exact: a BF16 value is the top half of an fp32 one. The kernel's results can
therefore differ slightly from tinygrad's own path, which first rounds BF16 weights to fp16.
"""

from __future__ import annotations

import functools
import math
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import _amd_load, warp_reduce
from tinygrad.llm.kernels.amd_gguf import _half
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

GGML_F16, GGML_BF16 = 1, 30
MAX_TOKEN_TILE = 8  # rows of activations served by one read of the weights
MAX_WAVES = 8  # waves per workgroup, which share one copy of the activations
CHUNK = 1024  # inputs per workgroup


def _unwrapped(u: UOp) -> UOp:
    """Strip views and float-to-float casts: the same unwrapping as tinygrad's `find_packed`."""
    while (
        u.op in (Ops.RESHAPE, Ops.STAGE)
        or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype))
        or (u.op is Ops.PERMUTE and (kept := [a for a in u.marg if u.src[0].shard_shape[a] != 1]) == sorted(kept))
    ):
        u = u.src[0]
    return u


def find_dense(decoded: Tensor) -> tuple[int, Tensor] | None:
    """(ggml type, weights as uint32 words) if `decoded` is exactly a BF16 or F16 GGUF tensor, possibly cast.

    The words are a zero-copy view of the model file's buffer: two weights per word, row-major.
    """
    n = decoded.numel()
    if decoded.uop.axis is not None or not isinstance(n, int):
        return None
    raw = next(
        (
            u
            for u in decoded.uop.toposort()
            if u.op in (Ops.SHRINK, Ops.BUFFER) and u.dtype == dtypes.uint8 and prod(u.shape) == 2 * n
        ),
        None,
    )
    if raw is None:
        return None
    target = _unwrapped(decoded.uop).key
    for ggml_type in (GGML_BF16, GGML_F16):
        if _unwrapped(ggml_data_to_tensor(Tensor(raw).flatten(), n, ggml_type).uop).key == target:
            break
    else:
        return None
    offset = raw.contiguous_view_offset()
    if offset is None or offset % 4 or raw.buf_uop.dtype != dtypes.uint8:
        return None
    return ggml_type, Tensor(raw).flatten().bitcast(dtypes.uint32).contiguous()


@functools.cache
def _dense_decode_kernel(
    out: UOp,
    words: UOp,
    x: UOp,
    *,
    out_features: int,
    in_features: int,
    half: bool,
    token_tile: int,
    waves: int,
    rows_per_wave: int,
) -> UOp:
    """out[token, row, chunk] = partial dot product of one weight row with one row of activations over one chunk.

    A workgroup handles `token_tile` rows of activations, `waves * rows_per_wave` weight rows and one chunk of
    CHUNK inputs. It first copies that chunk of the activations to shared memory (LDS), so they are read from
    global memory once per workgroup instead of once per wave. Each lane then takes 4 consecutive inputs out of
    every 128: its weight loads are contiguous across the wave, and its 16-byte LDS reads hit distinct banks.
    """
    T, W, R = token_tile, waves, rows_per_wave
    tokens, _, chunks = out.shape
    row_words, blocks = in_features // 2, out_features // (W * R)
    tile = UOp.range(int(tokens) // T * blocks, 0, AxisType.GLOBAL)
    chunk = UOp.range(int(chunks), 1, AxisType.GLOBAL)
    lane = UOp.range(32, 2, AxisType.LOCAL)
    wave = UOp.range(W, 3, AxisType.LOCAL)
    token0, row0, tid = tile // blocks * T, (tile % blocks) * (W * R) + wave * R, wave * 32 + lane
    zero = UOp.const(0, dtypes.float32)

    # stage x[token0:token0+T, chunk*CHUNK : (chunk+1)*CHUNK] in LDS, zero past the end of the row
    lds = UOp.alloc((T * CHUNK,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    staged = []
    for i in range(T * CHUNK // (W * 32)):
        e = tid + i * (W * 32)
        col = chunk * CHUNK + e % CHUNK
        value = x[token0 + e // CHUNK, col.minimum(in_features - 1)].load().float()
        staged.append(lds[e].store((col < in_features).where(value, zero)))
    lds = lds.after(*staged)

    def to_float(word: UOp) -> tuple[UOp, UOp]:
        # a word holds weights 2k (low 16 bits) and 2k+1 (high 16 bits)
        if half:
            return _half(word & 0xFFFF), _half(word >> 16)
        return (word << 16).bitcast(dtypes.float32), (word & 0xFFFF0000).bitcast(dtypes.float32)

    accs = [[zero] * T for _ in range(R)]
    for k in range(CHUNK // 128):
        elem = k * 128 + lane * 4  # this lane's 4 inputs within the chunk
        valid = chunk * CHUNK + elem < in_features
        xv = [_amd_load(lds[t * CHUNK + elem], 4) for t in range(T)]
        for r in range(R):
            row_base = (row0 + r) * row_words
            # past the end of the row the activations are zero: read the row's first words instead of the next row
            index = valid.where(row_base + (chunk * CHUNK + elem) // 2, row_base)
            wv = _amd_load(words[index], 2, stream=True)
            w = (*to_float(wv[0]), *to_float(wv[1]))
            for t in range(T):
                for j in range(4):
                    accs[r][t] = accs[r][t] + w[j] * xv[t][j]

    stores = [
        out[token0 + t, row0 + r, chunk.valid(lane.eq(0))].store(warp_reduce(accs[r][t], full_wave=True))
        for r in range(R)
        for t in range(T)
    ]
    name = f"linear_{'f16' if half else 'bf16'}_dense"
    return UOp.group(*stores).end(tile, wave, chunk, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))


def _layout(out_features: int, tokens: int) -> tuple[int, int, int]:
    """(token_tile, waves, rows_per_wave) for a layer and a token count."""
    token_tile = max(t for t in range(1, MAX_TOKEN_TILE + 1) if tokens % t == 0)
    waves = math.gcd(out_features, MAX_WAVES)
    rows = next(r for r in (4, 2, 1) if out_features % (waves * r) == 0)
    return token_tile, waves, rows


def dense_linear(layer: Any, x: Tensor) -> Tensor:
    """`layer(x)` for a Linear whose weights `find_dense` recognised, with a static token count."""
    ggml_type, words = layer._tinyllm_dense
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    tokens = int(prod(x.shape[:-1]))
    token_tile, waves, rows = _layout(out_features, tokens)
    out = Tensor.empty(tokens, out_features, math.ceil(in_features / CHUNK), dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(
        _dense_decode_kernel,
        out_features=out_features,
        in_features=in_features,
        half=ggml_type == GGML_F16,
        token_tile=token_tile,
        waves=waves,
        rows_per_wave=rows,
    )
    xs = x.reshape(tokens, in_features).contiguous()
    result = Tensor.custom_kernel(out, words, xs, fxn=fxn)[0].sum(-1).reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
