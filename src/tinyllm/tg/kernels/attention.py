"""Flash decode attention for several query rows at consecutive positions: the speculative verify (gfx11/gfx12).

The verify runs each row through decoding's own kernel (`tinygrad.llm.kernels.amd.amd_flash_attention_decode`)
so it computes exactly what one-token decoding computes (tinyllm.tg.speculative). Called once per row, that kernel
reads every cached key and value once per row: at 32k positions, 128 calls of 1.4 ms in an 8-row verify of the 27B,
60% of the cycle. Here one launch covers all rows. The row is a grid axis of its own, and each row's work is
upstream's partial and combine kernels unchanged, with that row's valid length (`valid_first + row`): the same
operations in the same order, so the result is the same bit for bit (`tools/attention_rows_check.py`). The blocks
of all rows for one chunk of keys are dispatched next to each other, so after the first reads the chunk comes from
the cache instead of memory.

Mirrors `_amd_flash_attention_decode_partial` and `_amd_flash_decode_combine` at the pinned commit; re-check on
every tinygrad bump.
"""

from __future__ import annotations

import functools
import math
from typing import Any

from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.llm.kernels.amd import LDS_PAD, LOG2E, WARP_SIZE, _reg, _unbind, _vec_load, warp_reduce
from tinygrad.uop.ops import AxisType, KernelInfo

BLOCK_N = 64  # keys per chunk, as decoding uses
PARTIALS = 48  # most chunk partials per row, as decoding uses
# range ids of the partial kernel's grid axes; the order decides which blocks are dispatched next to each other
GRID = {"head": 0, "chunk": 1, "row": 2}  # rows slowest: measured fastest (5.7 against 7.2 ms, 8 rows at 32k)


@functools.cache
def _partial_rows(out: UOp, stats: UOp, q: UOp, cache_kv: UOp, valid_first: Any, block_n: int, waves: int) -> UOp:
    valid_first = _unbind(valid_first)
    _, B, H_KV, _, D = cache_kv.shape
    _, H, M, _ = q.shape
    assert H % H_KV == 0 and D % WARP_SIZE == 0
    G, CHUNK, DPL, WAVES = H // H_KV, block_n, D // WARP_SIZE, waves
    PART = out.shape[2] // M
    assert CHUNK % WAVES == 0
    SEC = CHUNK // WAVES  # keys each wave scans independently
    last_chunks = (valid_first + (M - 1) + CHUNK - 1) // CHUNK
    live_chunks = min(last_chunks, PART) if isinstance(last_chunks, int) else last_chunks.minimum(PART)
    row = UOp.range(M, GRID["row"], AxisType.GLOBAL)
    block_bhkv = UOp.range(B * H_KV, GRID["head"], AxisType.GLOBAL)
    block_chunk = UOp.range(live_chunks, GRID["chunk"], AxisType.GLOBAL)
    lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
    valid_kv_len = valid_first + row  # this row's keys: decoding's valid length at its position
    total_chunks = (valid_kv_len + CHUNK - 1) // CHUNK
    b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
    # per-lane query fragments for every GQA head, kept packed in registers; unpacked at use
    qf = tuple(_vec_load(q[b, kv_head * G + h, row, lane * DPL], DPL) for h in range(G))
    zerof = UOp.const(0, dtypes.float)
    # Each block scans every PART-th chunk of its row, keeping an online softmax across rounds; a block past the
    # row's last chunk (the row has fewer than the last row) runs no round and its partial is never combined.
    chunk_round = UOp.range((total_chunks - block_chunk + PART - 1) // PART, 4)
    chunk_id = block_chunk + chunk_round * PART
    valids: list[UOp] = []
    scores: list[list[UOp]] = [[zerof] * G for _ in range(SEC)]
    vfrags: list[tuple[UOp, ...]] = [()] * SEC
    for j in range(SEC):
        key = chunk_id * CHUNK + wave * SEC + j
        valid = key < valid_kv_len
        valids.append(valid)
        kfrag = _vec_load(cache_kv[0, b, kv_head, key, lane * DPL], DPL)
        # V is prefetched in the score pass so both streams are in flight together
        vfrags[j] = tuple(valid.where(v, zerof) for v in _vec_load(cache_kv[1, b, kv_head, key, lane * DPL], DPL))
        for h in range(G):
            s = warp_reduce(
                sum((qf[h][i] * kfrag[i] for i in range(DPL)), UOp.const(0, dtypes.float)), full_wave=True
            ) * (1 / math.sqrt(D))
            scores[j][h] = valid.where(s, UOp.const(-1e30, dtypes.float))
    # A finite initial max keeps fully masked waves from computing exp(-inf - -inf).
    acc_reg, max_reg, sum_reg = _reg((G, DPL), 0), _reg((G,), -1e30), _reg((G,), 0)
    prev_acc, prev_max, prev_sum = acc_reg.after(chunk_round), max_reg.after(chunk_round), sum_reg.after(chunk_round)
    row_max = [functools.reduce(UOp.maximum, (scores[j][h] for j in range(SEC)), prev_max[h].load()) for h in range(G)]
    # Rescale the previous rounds to the new max, then accumulate this round's keys.
    alpha = [((prev_max[h].load() - row_max[h]) * LOG2E).exp2() for h in range(G)]
    accs = [[alpha[h] * prev_acc[h, i].load() for i in range(DPL)] for h in range(G)]
    row_sums = [alpha[h] * prev_sum[h].load() for h in range(G)]
    for j in range(SEC):
        for h in range(G):
            beta = valids[j].where(((scores[j][h] - row_max[h]) * LOG2E).exp2(), zerof)
            accs[h] = [a + beta * v for a, v in zip(accs[h], vfrags[j], strict=True)]
            row_sums[h] = row_sums[h] + beta
    update = UOp.group(
        acc_reg.store(UOp.stack(*(x for acc in accs for x in acc)).reshape(G, DPL)),
        max_reg.store(UOp.stack(*row_max)),
        sum_reg.store(UOp.stack(*row_sums)),
    ).end(chunk_round)
    acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
    # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
    acc_lds = UOp.alloc((WAVES, G, D + (LDS_PAD if G == SEC else 0)), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :, :D]
    ml_lds = UOp.alloc((WAVES, G, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
    lds_acc = acc_lds.reshape(WAVES, G, WARP_SIZE, DPL)
    # Normalize before fp16 to avoid overflow. Nonempty waves have sum >= 1; empty waves keep their zero accumulator.
    stores = [
        lds_acc[wave, h, lane].store((acc_reg[h].load() / sum_reg[h].load().maximum(1)).cast(dtypes.half))
        for h in range(G)
    ]
    stores += [
        ml_lds[wave, h, i].store(x) for h in range(G) for i, x in enumerate((max_reg[h].load(), sum_reg[h].load()))
    ]
    acc_lds, ml_lds = acc_lds.after(*stores), ml_lds.after(*stores)
    tid = wave * WARP_SIZE + lane
    slot = row * PART + block_chunk
    final_stores: list[UOp] = []
    for i in range(-(-G * D // (WAVES * WARP_SIZE))):
        flat = tid + i * WAVES * WARP_SIZE
        h, d = flat // D, flat % D
        top = functools.reduce(UOp.maximum, (ml_lds[w, h, 0].load() for w in range(WAVES)))
        # LDS holds normalized values; restore each wave's sum before combining.
        weights = (((ml_lds[w, h, 0].load() - top) * LOG2E).exp2() * ml_lds[w, h, 1].load() for w in range(WAVES))
        val = sum((x * acc_lds[w, h, d].load().float() for w, x in enumerate(weights)), zerof)
        oidx = out[b, kv_head * G + h, slot, d]
        if G * D % (WAVES * WARP_SIZE):
            oidx = out[b, (kv_head * G + h).valid(flat < G * D), slot, d]
        final_stores.append(oidx.store(val))
    hstat = tid
    top = functools.reduce(UOp.maximum, (ml_lds[w, hstat, 0].load() for w in range(WAVES)))
    total = sum(
        (((ml_lds[w, hstat, 0].load() - top) * LOG2E).exp2() * ml_lds[w, hstat, 1].load() for w in range(WAVES)), zerof
    )
    q_head = (kv_head * G + hstat).valid(hstat < G) if WAVES * WARP_SIZE > G else kv_head * G + hstat
    final_stores += [stats[b, q_head, slot, 0].store(top), stats[b, q_head, slot, 1].store(total)]
    return (
        UOp.group(*final_stores)
        .end(lane, wave, block_chunk, block_bhkv, row)
        .sink(arg=KernelInfo(name="flash_decode_rows_partial", opts_to_apply=()))
    )


@functools.cache
def _combine_rows(o: UOp, partial: UOp, stats: UOp, valid_first: Any) -> UOp:
    # one wave per (row, batch, head, 64-dim tile): every lane redundantly weights its row's chunks
    valid_first = _unbind(valid_first)
    B, H, M, D = o.shape
    PART = partial.shape[2] // M
    DT = 64 if D % 64 == 0 else WARP_SIZE  # dims per block
    assert D % DT == 0
    row = UOp.range(M, 0, AxisType.GLOBAL)
    block_bh, block_dt = UOp.range(B * H, 1, AxisType.GLOBAL), UOp.range(D // DT, 2, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, 3, axis_type=AxisType.LOCAL)
    b, h = block_bh // H, block_bh % H
    NPD = DT // WARP_SIZE  # output dims per lane
    dims = tuple(block_dt * DT + lane * NPD + i for i in range(NPD))
    live = ((valid_first + row + BLOCK_N - 1) // BLOCK_N).minimum(PART)  # this row's chunks, as decoding combines
    base = row * PART

    def iloop(ph: UOp, val: float) -> UOp:
        return ph.store(ph.const_like(val))

    chunk = UOp.range(live, 100)
    chunk_max = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
    chunk_max_i = chunk_max.after(iloop(chunk_max, -math.inf))
    update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[b, h, base + chunk, 0].load())).end(chunk)
    chunk_max = chunk_max_i.after(update0)
    chunk2 = UOp.range(live, 101)
    acc = UOp.alloc((NPD,), dtypes.float, addrspace=AddrSpace.REG)
    weight_sum = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
    acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, 0))
    w = ((stats[b, h, base + chunk2, 0].load() - chunk_max) * LOG2E).exp2()
    update1 = UOp.group(
        *[
            acc_i[i].store(acc_i.after(chunk2)[i].load() + w * partial[b, h, base + chunk2, d].load())
            for i, d in enumerate(dims)
        ],
        weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w * stats[b, h, base + chunk2, 1].load()),
    ).end(chunk2)
    acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
    inv = 1 / weight_sum[0].load()
    return (
        UOp.group(*[o[b, h, row, d].store(acc[i].load() * inv) for i, d in enumerate(dims)])
        .end(lane, block_dt, block_bh, row)
        .sink(arg=KernelInfo(name="flash_decode_rows_combine", opts_to_apply=()))
    )


def supported(q: Tensor, cache_kv: Tensor) -> bool:
    """Whether decoding takes its flash kernel for these shapes (`tinygrad.llm.kernels.amd.flash_attention`)."""
    D, N, group = int(q.shape[3]), int(cache_kv.shape[3]), int(q.shape[1]) // int(cache_kv.shape[2])
    return D % 32 == 0 and D & (D - 1) == 0 and N % 64 == 0 and group * ((D + LDS_PAD) * 2 + 8) <= 65536


def flash_decode_rows(q: Tensor, cache_kv: Tensor, valid_first: Any, max_kv_len: int) -> Tensor:
    """Attention of `q` (B, H, M, D), row j over the first `valid_first + j` cached positions of `cache_kv`
    (2, B, H_KV, N, D) half: (B, H, M, D) float32, row j equal bit for bit to decoding's call for that row."""
    if isinstance(valid_first, UOp):  # carry the length binding into the graph, as decoding does
        cache_kv = Tensor(cache_kv.uop.after(valid_first))
    B, H, M, D = q.shape
    chunks, axis = min(PARTIALS, max_kv_len // BLOCK_N), q.uop.axis
    partial = Tensor.empty(B, H, M * chunks, D, dtype="float32", device=q.device, axis=axis)
    stats = Tensor.empty(B, H, M * chunks, 2, dtype="float32", device=q.device, axis=axis)
    waves, group = 16, H // cache_kv.shape[2]
    while waves * group * ((D + LDS_PAD) * 2 + 8) > 65536:
        waves //= 2
    assert waves > 0, "attention head group exceeds shared memory capacity"
    fxn = functools.partial(_partial_rows, valid_first=valid_first, block_n=BLOCK_N, waves=waves)
    partial, stats = Tensor.custom_kernel(partial, stats, q.half(), cache_kv, fxn=fxn)[:2]
    out = Tensor.empty(B, H, M, D, dtype="float32", device=q.device, axis=axis)
    return Tensor.custom_kernel(out, partial, stats, fxn=functools.partial(_combine_rows, valid_first=valid_first))[0]
