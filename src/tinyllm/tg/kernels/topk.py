"""Top-k of a long row on the GPU, for sampling (k up to 32 over Qwen3.8's 248,320 logits).

tinygrad's `Tensor.topk` sorts the whole row: 95 ms for 248k values on Strix Halo, longer than a decode step.
Taking the maximum k times costs 4.6 ms for k = 20. Here a bitonic sort in local memory runs in passes: each
workgroup sorts one block of the row (values with their indices) and keeps its k largest, and the next pass does
the same over those candidates, until one block remains (three passes for 248,320 values). The k largest of the
row are always among the k largest of the blocks that hold them, so the result is exact; equal values keep the
lower index first, as `argmax` does.
"""

from __future__ import annotations

import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

# values sorted per workgroup in each pass. Larger blocks need fewer passes but unroll into much longer programs:
# with 4,096 the 27B server took 62 s longer to start and 4.8 GB more memory, for the same 0.14 ms per call.
BLOCK = 1024
THREADS = 256


@functools.cache
def _sort_block_kernel(out_v: UOp, out_i: UOp, x: UOp, ids: UOp | None = None, *, n: int, k: int, block: int) -> UOp:
    """For each block of `block` values of x[:n] (missing values count as -inf): its k largest, descending, with
    their indices (positions in x, or `ids[position]` when given)."""
    groups = math.ceil(n / block)
    wg = UOp.range(groups, 0, AxisType.GLOBAL)
    tid = UOp.range(THREADS, 1, AxisType.LOCAL)
    vals = UOp.alloc((block,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    idxs = UOp.alloc((block,), dtypes.int32, addrspace=AddrSpace.LOCAL)
    stores = []
    for i in range(block // THREADS):
        e = tid + i * THREADS
        g = wg * block + e
        inside = g < n
        safe = inside.where(g, 0)
        value = inside.where(x[safe].load(), UOp.const(-math.inf, dtypes.float32))
        index = ids[safe].load() if ids is not None else safe.cast(dtypes.int32)
        stores += [vals[e].store(value), idxs[e].store(inside.where(index, UOp.const(2**31 - 1, dtypes.int32)))]
    vals, idxs = vals.after(*stores), idxs.after(*stores)
    size = 2
    while size <= block:  # bitonic sort, descending; each stage swaps disjoint pairs, a barrier between stages
        stride = size // 2
        while stride >= 1:
            stage = []
            for p in range(block // 2 // THREADS):
                pair = tid + p * THREADS
                a = (pair // stride) * (2 * stride) + pair % stride
                b = a + stride
                va, vb, ia, ib = vals[a].load(), vals[b].load(), idxs[a].load(), idxs[b].load()
                b_first = (va < vb) | (va.eq(vb) & (ia > ib))  # b belongs before a in descending order
                descending = ((a // size) % 2).eq(0) if size < block else UOp.const(True, dtypes.bool)
                swap = descending.where(b_first, b_first.logical_not() & (va.eq(vb) & ia.eq(ib)).logical_not())
                stage += [
                    vals[a].store(swap.where(vb, va)),
                    vals[b].store(swap.where(va, vb)),
                    idxs[a].store(swap.where(ib, ia)),
                    idxs[b].store(swap.where(ia, ib)),
                ]
            vals, idxs = vals.after(*stage), idxs.after(*stage)
            stride //= 2
        size *= 2
    j = tid.minimum(k - 1)
    writes = [
        out_v[(wg * k + tid).valid(tid < k)].store(vals[j].load()),
        out_i[(wg * k + tid).valid(tid < k)].store(idxs[j].load()),
    ]
    name = f"topk{k}_block{block}" + ("_ids" if ids is not None else "")
    return UOp.group(*writes).end(wg, tid).sink(arg=KernelInfo(name=name, opts_to_apply=()))


def topk(row: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """The k largest values of a 1-D float32 tensor, descending, and their positions (int32)."""
    n = int(row.shape[0])
    if not 1 <= k <= THREADS or k > n:
        raise ValueError(f"k={k} out of range for {n} values")
    values, ids = row.float().contiguous(), None
    while True:  # each pass keeps the k largest of every block, until one block remains
        groups = math.ceil(n / BLOCK)
        out_v = Tensor.empty(groups * k, dtype=dtypes.float32, device=row.device)
        out_i = Tensor.empty(groups * k, dtype=dtypes.int32, device=row.device)
        fxn = functools.partial(_sort_block_kernel, n=n, k=k, block=BLOCK)
        sources = (values,) if ids is None else (values, ids)
        values, ids = Tensor.custom_kernel(out_v, out_i, *sources, fxn=fxn)[:2]
        if groups == 1:
            return values, ids
        n = groups * k
