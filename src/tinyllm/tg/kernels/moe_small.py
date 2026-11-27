"""Expert (MoE) layers stored as Q5_0, Q5_1 or Q2_0: packed-weight kernels, and Q2_0 decoding for tinygrad.

tinygrad's packed MoE kernels (`tinygrad.llm.kernels.amd_gguf.moe_linear`) cover Q8_0, the K-quants, the IQ
types and MXFP4. unsloth's Qwen3.8-Flash-Next files store most expert down projections (640 -> 2,560) in other
types: Q5_1 in `UD-Q4_K_XL` (43 layers, 27 GB), Q5_0 (33 layers) and Q2_0 (2) in `UD-IQ4_XS`. tinygrad runs them
through generic code that decodes the gathered experts' weights: 34 ms of a 71 ms decode step
(docs/04-qwen3.8-flash-next.md, section 7). The kernels here keep the layout of tinygrad's decode kernels (one
wave per output row and up to 1,024 inputs, one lane per group of 32 weights) but not their integer dot product:
tinygrad rounds the activations to int8 per group of 32 first, which moved these layers' logits away from
llama.cpp's (section 11), while decoding a token reads each weight once and is bound by memory, not arithmetic.
So each lane multiplies its 32 codes, as floats, by fp32 activations.

Blocks (ggml `block_q5_0`, `block_q5_1`, `block_q2_0`):
  Q5_0, 32 weights in 22 bytes: fp16 d, a 32-bit qh, 16 bytes of nibbles (weight j in the low nibble of byte j,
      weight j + 16 in the high one); bit j of qh is weight j's fifth bit. w = d * (q - 16).
  Q5_1, 32 weights in 24 bytes: fp16 d, fp16 m, qh, nibbles as above. w = d * q + m.
  Q2_0, 64 weights in 18 bytes: fp16 d, then four 2-bit codes per byte in order. w = d * (q - 1).
So a group's sum over its activations x is d * (dot - c * sum(x)) for Q5_0 (c = 16) and Q2_0 (c = 1), and
d * dot + m * sum(x) for Q5_1, with dot the sum of code * activation.

tinygrad cannot decode Q2_0 (it appeared with llama.cpp b12675), so `install_decoders` adds it to tinygrad's
GGUF reader and decoder before a file is parsed.
"""

from __future__ import annotations

import functools
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm import gguf as tg_gguf
from tinygrad.llm.kernels.amd import _amd_load
from tinygrad.llm.kernels.amd_gguf import _decode_linear, _half, _load_u32
from tinygrad.uop.ops import Ops

Q5_0, Q5_1, Q2_0 = 6, 7, 42
NAMES = {Q5_0: "q5_0", Q5_1: "q5_1", Q2_0: "q2_0"}
BLOCK = {Q5_0: (32, 22), Q5_1: (32, 24), Q2_0: (64, 18)}  # weights and bytes per block
# raw words the kernels index: Q5_1 blocks are 24 bytes (u32-aligned), the others only halfword-aligned
WORD = {Q5_0: dtypes.uint16, Q5_1: dtypes.uint32, Q2_0: dtypes.uint16}


def _decode_q2_0(blocks: Tensor) -> Tensor:
    """(blocks, 18) bytes -> (blocks, 64) float32: d * (q - 1), four codes per byte in order (ggml's layout)."""
    d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32)
    codes = blocks[:, 2:].unsqueeze(-1).div(Tensor.const((1, 4, 16, 64), blocks.dtype), rounding_mode="trunc")
    q = codes.bitwise_and(3).flatten(-2).cast(dtypes.int8)
    return (q - 1).cast(dtypes.float32) * d


def install_decoders() -> None:
    """Teach tinygrad's GGUF reader and decoder Q2_0 (once)."""
    if Q2_0 in tg_gguf._GGML_QUANT:
        return
    tg_gguf._GGML_QUANT[Q2_0] = BLOCK[Q2_0]
    original = tg_gguf.ggml_data_to_tensor

    def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
        if ggml_type != Q2_0:
            return original(t, n, ggml_type)
        elems, nbytes = BLOCK[Q2_0]
        blocks = t[: (n // elems) * nbytes].reshape((-1, nbytes)).contiguous()
        return _decode_q2_0(blocks)  # (blocks, 64), as tinygrad's decoders return; the loader reshapes

    ggml_data_to_tensor._tinyllm_original = original  # type: ignore[attr-defined]
    tg_gguf.ggml_data_to_tensor = ggml_data_to_tensor


def find_small(decoded: Tensor, in_features: int) -> tuple[int, Tensor] | None:
    """(ggml type, raw words) if `decoded` is exactly tinygrad's decoding of Q5_0, Q5_1 or Q2_0 bytes, rows of
    `in_features` (as `amd_gguf.find_packed` does for the types tinygrad's kernels serve)."""
    if decoded.uop.axis is not None:
        return None
    sizes = {t: decoded.numel() // e * b for t, (e, b) in BLOCK.items() if in_features % e == 0}
    raw = next(
        (
            u
            for u in decoded.uop.toposort()
            if u.op in (Ops.SHRINK, Ops.BUFFER) and u.dtype == dtypes.uint8 and prod(u.shape) in sizes.values()
        ),
        None,
    )
    if raw is None:
        return None

    def unwrapped(u: UOp) -> UOp:
        while u.op in (Ops.RESHAPE, Ops.STAGE) or (
            u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)
        ):
            u = u.src[0]
        return u

    for ggml_type, size in sizes.items():
        if size != prod(raw.shape):
            continue
        expected = tg_gguf.ggml_data_to_tensor(Tensor(raw).flatten(), int(decoded.numel()), ggml_type)
        if unwrapped(decoded.uop).key == unwrapped(expected.uop).key:
            break
    else:
        return None
    word = WORD[ggml_type]
    offset = raw.contiguous_view_offset()
    if offset is None or offset % word.itemsize or raw.buf_uop.dtype != dtypes.uint8:
        return None
    return ggml_type, Tensor(raw).flatten().bitcast(word).contiguous()


def _codes(raw: UOp, base: UOp, group: UOp, ggml_type: int) -> list[UOp]:
    """The 32 codes of one group of 32 weights, in order, as uint32."""
    if ggml_type == Q2_0:  # bytes 2 + 8 * half on, four 2-bit codes a byte: weight j at bit 2 (j % 16) of word j // 16
        half = group % 2
        words = [_load_u32(raw, base, 2 + half * 8 + 4 * w) for w in range(2)]
        return [(words[j // 16] >> (2 * (j % 16))) & 3 for j in range(32)]
    if ggml_type == Q5_1:  # u32 words: d | m, qh, then 4 words of nibbles
        qh, qs = raw[base + 1].cast(dtypes.uint32), [raw[base + 2 + w].cast(dtypes.uint32) for w in range(4)]
    else:  # Q5_0, halfwords: d, qh (bytes 2..5), then nibbles (bytes 6..21)
        qh, qs = _load_u32(raw, base, 2), [_load_u32(raw, base, 6 + 4 * w) for w in range(4)]
    # weight j: the low nibble of byte j for j < 16, the high nibble of byte j - 16 after; bit j of qh on top
    return [((qs[(j % 16) // 4] >> (8 * (j % 4) + 4 * (j // 16))) & 15) | (((qh >> j) & 1) << 4) for j in range(32)]


@functools.cache
def _moe_small_kernel(
    out: UOp, raw: UOp, x: UOp, sel: UOp, out_features: int, in_features: int, ggml_type: int, x_share: int
) -> UOp:
    # out rows are (token, slot): the slot's expert reads its weight rows, x_share slots share an activation row
    elems, nbytes = BLOCK[ggml_type]
    words_per_block = nbytes // raw.dtype.itemsize
    expert_words = out_features * in_features // elems * words_per_block
    experts = raw.shape[0] // expert_words

    def group_dot(row: UOp, output: UOp, group: UOp) -> UOp:
        expert = sel[row].cast(dtypes.weakint).maximum(0).minimum(experts - 1)
        token = row // x_share
        block = group // 2 if ggml_type == Q2_0 else group
        base = expert * expert_words + (output * in_features // elems + block) * words_per_block
        xv = [_amd_load(x[token, group, 4 * k], 4) for k in range(8)]
        xs = [xv[j // 4][j % 4] for j in range(32)]
        # four partial sums of code * activation, so the multiply-adds do not wait on each other
        parts = [UOp.const(0, dtypes.float32)] * 4
        for j, code in enumerate(_codes(raw, base, group, ggml_type)):
            parts[j % 4] = parts[j % 4] + code.cast(dtypes.float32) * xs[j]
        dot, xsum = (parts[0] + parts[1]) + (parts[2] + parts[3]), functools.reduce(lambda a, b: a + b, xs)
        if ggml_type == Q5_1:
            dm = raw[base]
            return _half(dm & 0xFFFF) * dot + _half(dm >> 16) * xsum
        return _half(raw[base]) * (dot - (16 if ggml_type == Q5_0 else 1) * xsum)

    return _decode_linear(out, out_features, in_features // 32, group_dot, "moe_" + NAMES[ggml_type])


def moe_small_linear(layer: Any, found: tuple[int, Tensor], sel: Tensor, x: Tensor) -> Tensor:
    """`ExpertWeights.__call__` for these types: sel (B, T, k), x (B, T, 1 or k, in) -> (B, T, k, out)."""
    if not isinstance(T := sel.shape[1], int):  # symbolic token count: run on the padded maximum
        padded = moe_small_linear(layer, found, sel.pad_to(sel.max_shape, value=-1), x.pad_to(x.max_shape))
        return padded[..., :T, :, :]
    ggml_type, raw = found
    B, T, slots = (int(s) for s in sel.shape[-3:])
    in_features, x_slots = int(x.shape[-1]), int(x.shape[-2])
    out_features = int(layer.weight.shape[1])
    xf = x.float().reshape(B * T * x_slots, in_features // 32, 32).contiguous()
    out = Tensor.empty(B * T * slots, out_features, (in_features + 1023) // 1024, dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(
        _moe_small_kernel,
        out_features=out_features,
        in_features=in_features,
        ggml_type=ggml_type,
        x_share=slots // x_slots,
    )
    result = Tensor.custom_kernel(out, raw, xf, sel.cast(dtypes.int32).flatten(), fxn=fxn)[0].sum(-1)
    return result.reshape(B, T, slots, out_features)
