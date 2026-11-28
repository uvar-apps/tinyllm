"""Expert selection for prefill: each token's k largest router logits and their weights, in one kernel (gfx11).

tinygrad's `pairwise_topk` ranks every logit of a row against all the others, 512 x 512 comparisons a row for
Qwen3.8-Flash-Next's 512 experts, and the gather and the softmax that follow are generic kernels too: 0.38 ms and
twice 0.13 ms a block at 2,048 rows, 31 ms of a 2,048-token prefill call (docs/04-qwen3.8-flash-next.md section 14).

Here a wave takes a row. Each lane holds n / 32 of its logits as integer keys that order as the floats do, and
k rounds each take the wave's largest remaining key: the lane's own maximum, the wave's maximum of those, then the
lowest index among the lanes that hold it. That is `pairwise_topk`'s order (larger first, equal values by lower
index; -0.0 counts as 0.0), and the outputs keep its layout: the selected experts ascending, the largest last. The
weights are the softmax over the selected logits, summed in the same order; the caller applies the model's scaling
factor, so what follows is the expression it was.

Integer keys make the order total, so a row of padding (any bits, NaN included) still selects k experts in range.
"""

from __future__ import annotations

import functools
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.amd import (
    WARP_SIZE,
    _amd_load,
    _is_cdna,
    _is_rdna4,
    amd_custom_kernels_supported,
    warp_reduce,
)
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

ACTIVE = True
WEIGHTS = True  # False (checks): the caller computes the weights from the selection with tinygrad's expressions
MIN_ROWS = 128  # smaller calls keep tinygrad's kernels (decoding, and the calls the golden records are made of)
MAX_PER_LANE = 32  # logits a lane holds: up to 1,024 experts
_LOW31 = 0x7FFFFFFF
_TAKEN = -(2**31)


def _key(value: UOp) -> UOp:
    """A float32 as an int32 that orders the same way, -0.0 as 0.0."""
    bits = value.bitcast(dtypes.int32)
    key = (bits < 0).where(bits ^ _LOW31, bits)
    return key.eq(-1).where(UOp.const(0, dtypes.int32), key)


def _value(key: UOp) -> UOp:
    return (key < 0).where(key ^ _LOW31, key).bitcast(dtypes.float32)


@functools.cache
def _select_kernel(sel: UOp, probs: UOp, logits: UOp, *, k: int) -> UOp:
    """logits (tokens, n) float32 -> sel (tokens * k,) int32 and probs (tokens * k,) float32, a token's k entries
    ascending by logit."""
    tokens, n = int(logits.shape[0]), int(logits.shape[1])
    per = n // WARP_SIZE
    assert n % (WARP_SIZE * 8) == 0 and per <= MAX_PER_LANE and 1 <= k <= WARP_SIZE
    token = UOp.range(tokens, 0, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
    loads = [_amd_load(logits[token, lane * per + 8 * i], 8) for i in range(per // 8)]
    keys = [_key(vec[j]) for vec in loads for j in range(8)]
    own = [lane * per + e for e in range(per)]
    taken = UOp.const(_TAKEN, dtypes.int32)
    picked: list[tuple[UOp, UOp]] = []  # the wave's (key, index), largest first
    for _ in range(k):
        best, best_e = keys[0], UOp.const(0, dtypes.int32)
        for e in range(1, per):  # strictly greater: the lowest index of equal keys stays
            greater = keys[e] > best
            best, best_e = greater.where(keys[e], best), greater.where(UOp.const(e, dtypes.int32), best_e)
        top = warp_reduce(best, Ops.MAX, full_wave=True)
        # among the lanes that hold the wave's maximum, the lowest index: the largest of n - 1 - index
        behind = best.eq(top).where((n - 1) - (lane * per + best_e), UOp.const(-1, dtypes.int32))
        index = (n - 1) - warp_reduce(behind, Ops.MAX, full_wave=True)
        picked.append((top, index))
        keys = [own[e].eq(index).where(taken, keys[e]) for e in range(per)]
    values = [_value(key) for key, _ in picked]
    exps = [(v - values[0]).exp() for v in values]
    total = exps[k - 1]
    for r in range(k - 2, -1, -1):  # in the outputs' order: ascending by logit
        total = total + exps[r]
    stores = []
    for j in range(k):  # entry j is the (k - j)-th largest; lane j stores it
        place = (token * k + j).valid(lane.eq(j))
        stores.append(sel[place].store(picked[k - 1 - j][1]))
        stores.append(probs[place].store(exps[k - 1 - j] / total))
    return UOp.group(*stores).end(lane, token).sink(arg=KernelInfo(name=f"moe_select_top{k}", opts_to_apply=()))


def serves(logits: Tensor, k: int) -> bool:
    """Whether `select` takes these router logits: a fixed number of rows, whole lanes of 8 logits, on gfx11."""
    shape = logits.shape
    if not ACTIVE or not all(isinstance(s, int) for s in shape) or not isinstance(logits.device, str):
        return False
    if not amd_custom_kernels_supported(logits.device) or _is_rdna4(logits.device) or _is_cdna(logits.device):
        return False
    rows, n = int(shape[0]) * int(shape[1]), int(shape[-1])
    return rows >= MIN_ROWS and n % (WARP_SIZE * 8) == 0 and n // WARP_SIZE <= MAX_PER_LANE and 1 <= k <= WARP_SIZE


def select(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """For logits (B, T, n): sel (B, T, k) int32, the k largest logits' indices in `pairwise_topk`'s order, and
    probs (B, T, k) float32, their softmax."""
    B, T, n = (int(s) for s in logits.shape)
    flat: Any = logits.float().contiguous().reshape(B * T, n)
    sel = Tensor.empty(B * T * k, dtype=dtypes.int32, device=logits.device)
    probs = Tensor.empty(B * T * k, dtype=dtypes.float32, device=logits.device)
    fxn = functools.partial(_select_kernel, k=k)
    sel, probs = Tensor.custom_kernel(sel, probs, flat, fxn=fxn)[:2]
    return sel.reshape(B, T, k), probs.reshape(B, T, k)
