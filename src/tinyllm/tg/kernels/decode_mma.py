"""Decode attention on the matrix cores, split over the keys, for one or several rows (gfx11/gfx12).

Decoding's kernel (`tinygrad.llm.kernels.amd.amd_flash_attention_decode`) holds 8 of a key's 256 dimensions per
lane and reduces across the wave for every score. It reads the cache at about 140 GB/s for one row. For the K + 1
rows of a speculative verify its cost grows with rows times keys even when each key is read once
(`tinyllm.tg.kernels.attention`, research findings 5.10).

Here a block takes one KV head and up to `MAX_ROWS` rows, four waves per row. A row's slots are its query heads of
the KV head (6 on Qwen3.8-27B), padded to a 16-row matrix tile. Scores and the weighted sum of values are WMMA
products, following the tile loop of upstream's prefill kernel (`_amd_flash_attention`). The key tiles are dealt
out to `SPLITS` blocks (tile t to block t mod SPLITS), and a block reads each tile once for all its rows. Each row
keeps an online softmax over the block's tiles and writes its unnormalized sum, maximum and total. A second kernel
combines the blocks of each (row, head).

A row's work depends only on that row: its queries, its length, and the keys and values before that length. Keys
past it only produce scores that are masked out, and the values past it are zeroed in registers before the row's
product. The zeroing has to be per row: the matrix units do not add a zero probability times a value as an exact
zero, so values past a row's length that differ (the next verify rows' entries, or stale ones when decoding) would
change its last bits. Tiles past a row's length that its block scans for later rows leave its sums unchanged.

So a row gives the same result decoded alone (one row, the query in local memory) and verified among several,
bit for bit, provided decoding takes this kernel too. The compiler's choice of fused multiply-adds can still differ
between builds of different shapes; two waves per row instead of four did (by one ulp, past 3,072 keys). Every
shape the server builds (1, 2 and 4 rows; the 27B's and the 0.8B's heads) is checked by `tools/decode_mma_check.py`
in the GPU tests; a build that agrees at the checked positions agrees at all, since positions only change loop
counts and masks.

Mirrors `_amd_flash_attention` at the pinned commit for the tile loop; re-check on every tinygrad bump.
"""

from __future__ import annotations

import functools
import math
from typing import Any

from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.llm.kernels.amd import (
    BLOCK_N,
    LANES_PER_WAVE_M,
    LANES_PER_WAVE_N,
    LDS_PAD,
    LOG2E,
    WARP_SIZE,
    WMMA_ACC,
    WMMA_ARG,
    WMMA_K,
    WMMA_M,
    WMMA_N,
    _is_cdna,
    _is_rdna4,
    _reg,
    _unbind,
    _wmma_fragment,
    amd_custom_kernels_supported,
    warp_reduce,
)
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve

SPLITS = 48  # blocks a row's key tiles are dealt to, per KV head
SLOTS = WMMA_M  # query rows of a block: one row's heads of one KV head, padded to a matrix tile
WAVES_N = 4  # waves per row: they share out the output dimensions (each output's arithmetic is the same for any)
MAX_ROWS = 4  # rows in one block; more, or a count that is not a power of two, take several launches
Q_LOCAL = 1  # blocks of at most this many rows copy the query slots to local memory once (it fits)
# range ids of the grid axes (2 and 3 are the waves' local axes)
GRID = {"head": 0, "split": 1}


@functools.cache
def _partial(part: UOp, stats: UOp, q: UOp, cache: UOp, valid_first: Any, rows: int, rdna4: bool) -> UOp:
    valid_first = _unbind(valid_first)
    BKV, RS, D = q.shape  # (batch * KV heads, rows * SLOTS, head size) half
    _, B, H_KV, N, cache_dim = cache.shape
    splits = part.shape[1]
    k, v = cache[0].reshape(B * H_KV, N, cache_dim), cache[1].reshape(B * H_KV, N, cache_dim)
    assert isinstance(D, int) and D % WMMA_K == 0 and D % (WAVES_N * LANES_PER_WAVE_N) == 0 and rows * SLOTS == RS
    row_threads = WARP_SIZE * WAVES_N
    threads = rows * row_threads
    assert BLOCK_N * D % threads == 0, "rows must divide the key tile's loads (a power of two up to MAX_ROWS)"
    TM, TN = SLOTS // LANES_PER_WAVE_M, BLOCK_N // LANES_PER_WAVE_N
    TD, SCALE = D // (WAVES_N * LANES_PER_WAVE_N), 1 / math.sqrt(D)
    last = valid_first + (rows - 1)  # the last row's length: the block scans its tiles, the others mask the rest
    tiles = (last + BLOCK_N - 1) // BLOCK_N
    live = min(tiles, splits) if isinstance(tiles, int) else tiles.minimum(splits)
    block_bh, split = UOp.range(BKV, GRID["head"], AxisType.GLOBAL), UOp.range(live, GRID["split"], AxisType.GLOBAL)
    # four waves per row: each row's work is what a block of `rows=1` does, the rows share the key and value tiles
    row, wave_n = UOp.range(rows, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL)
    lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
    tid, rtid = (row * WAVES_N + wave_n) * WARP_SIZE + lane, wave_n * WARP_SIZE + lane
    lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
    valid = valid_first + row  # this row attends to its own position and every one before
    KV_ELEMS = BLOCK_N * D // threads
    if rows <= Q_LOCAL:  # every row's query slots copied to local memory once, before the tiles
        Q_ELEMS = rows * SLOTS * D // threads
        Q_lds = UOp.alloc((rows * SLOTS, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
        q_rows = q.reshape(BKV, rows * SLOTS, D)[block_bh].reshape(threads, Q_ELEMS)
        Q_lds = Q_lds.after(Q_lds.reshape(threads, Q_ELEMS)[tid].store(q_rows[tid]))
        qv = Q_lds.reshape(rows, SLOTS, D)[row]
    else:  # this row's query slots, read from memory at each tile
        qv = q.reshape(BKV, rows, SLOTS, D)[block_bh, row]
    k, v = k[block_bh], v[block_bh]
    KV_lds = UOp.alloc((BLOCK_N, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
    acc, m_i, l_i = _reg((TM, TD), 0), _reg((TM,), -math.inf), _reg((TM,), 0)
    rnd = UOp.range((tiles - split + splits - 1) // splits, 100)  # this block's tiles: split, split + SPLITS, ...
    tile = split + rnd * splits
    load_k = UOp.range(KV_ELEMS, 90)
    kval = k.reshape(N * D)[tile * BLOCK_N * D + tid * KV_ELEMS + load_k]
    K_store = KV_lds.reshape(threads, KV_ELEMS)[tid, load_k].store(kval).end(load_k)
    KV_lds_k = KV_lds.after(K_store)
    S_reg = _reg((TM, TN), 0, rnd)
    k_qk, tm1, tn1 = UOp.range(D // WMMA_K, 101), UOp.range(TM // WMMA_ACC, 200), UOp.range(TN, 201)
    S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
    q_frag = qv.after(rnd).reshape(TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[tm1, lane_n, k_qk]
    k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
    q_frag, k_frag = (_wmma_fragment(f, lane, rdna4) for f in (q_frag, k_frag))
    qk = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk).barrier()
    S_reg = S_reg.after(qk, S_reg.store(S_reg * SCALE))
    rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
    k_idx = tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
    masked = (k_idx < valid).where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))
    S_reg = S_reg.after(S_reg[rm, rn].store(masked).end(rm, rn))
    m_ij, rm2 = _reg((TM,), -math.inf, rnd), UOp.range(TN, 261)
    m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
    ri_w = UOp.range(TM, 270)
    m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], Ops.MAX)).end(ri_w))
    tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
    S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
    p_local, ri_ws = _reg((TM,), 0, rnd), UOp.range(TM, 295)
    p_sum = p_local.after(p_local[ri_ws].store(sum(warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN))).end(ri_ws))
    # one probability tile per row; its four waves computed the same scores and store the same values
    P_lds = UOp.alloc((rows, SLOTS, BLOCK_N), dtypes.half, addrspace=AddrSpace.LOCAL)
    # gfx11 distributes even/odd rows between half-waves; gfx12 distributes the low/high eight rows.
    row_shape = (LANES_PER_WAVE_M, TM) if rdna4 else (TM, LANES_PER_WAVE_M)
    row_lane, row_elem = (2, 3) if rdna4 else (3, 2)
    P_write = (
        P_lds[row]
        .reshape(1, 1, *row_shape, TN, LANES_PER_WAVE_N)
        .permute(1, 0, row_lane, 5, row_elem, 4)
        .reshape(WARP_SIZE, TM, TN)
    )
    P_store = P_write[lane].store(S_reg.cast(dtypes.half))
    beta_i, ri4, rj4 = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG), UOp.range(TM, 330), UOp.range(TD, 331)
    m_new = m_i[ri4].maximum(m_ij[ri4])
    alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
    correction = UOp.group(
        acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
        l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
        m_i[ri4].store(m_new),
        beta_i[ri4].store(beta_val),
    ).end(ri4)
    acc, l_i, m_i = acc.after(correction), l_i.after(correction), m_i.after(correction)
    beta_i = beta_i.after(correction)
    V_lds = UOp.alloc((D, BLOCK_N + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
    V_copy, load_v = V_lds.after(qk).permute(1, 0), UOp.range(KV_ELEMS, 390)
    v_pos = tile * BLOCK_N + (tid * KV_ELEMS + load_v) // D
    # values past the last row's length may be unwritten: zeroed here, and past each row's own length below
    vval = (v_pos < last).where(v.reshape(N * D)[tile * BLOCK_N * D + tid * KV_ELEMS + load_v], 0)
    V_store = V_copy.reshape(threads, KV_ELEMS)[tid, load_v].store(vval).end(load_v)
    P_lds, V_lds = P_lds.after(P_store, V_store), V_lds.after(P_store, V_store)
    pv_acc = _reg((TM, TD), 0, rnd)
    k_pv, tm2, tn2 = UOp.range(BLOCK_N // WMMA_K, 400), UOp.range(TM // WMMA_ACC, 401), UOp.range(TD, 402)
    pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
    p_frag = P_lds[row].reshape(TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[tm2, lane_n, k_pv]
    v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
    # this row's values: those past its own length zeroed in registers, as a one-row block zeroes them on loading
    # (the matrix units do not add a zero probability times a value as an exact zero)
    v_row, vi = UOp.alloc((WMMA_K,), dtypes.half, addrspace=AddrSpace.REG), UOp.range(WMMA_K, 403)
    v_key = tile * BLOCK_N + k_pv * WMMA_K + vi
    v_row = v_row.after(v_row[vi].store((v_key < valid).where(v_frag[vi], v_frag[vi].const_like(0))).end(vi))
    p_frag, v_row = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_row))
    pv_done = pv_frag.store(UOp.wmma(p_frag, v_row, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
    pv_acc = pv_acc.after(pv_done)
    ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
    rnd_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).end(rnd)
    acc, l_i, m_i = acc.after(rnd_end), l_i.after(rnd_end), m_i.after(rnd_end)
    # this block's unnormalized sum, maximum and total for each slot: the combine kernel weighs the blocks
    out = part.reshape(BKV, splits, rows, SLOTS, D)[block_bh, split, row]
    out = (
        out.reshape(1, *row_shape, WAVES_N, TD, LANES_PER_WAVE_N)
        .permute(0, 3, row_lane - 1, 5, row_elem - 1, 4)
        .reshape(row_threads, TM, TD)
    )
    rs = UOp.range(TM, 420)
    slot = row * SLOTS + (lane_m * TM + rs if rdna4 else rs * LANES_PER_WAVE_M + lane_m)
    # every lane of a half-wave holds the same maximum and total for its slots: duplicate stores of one value
    stat_stores = UOp.group(
        stats[block_bh, split, slot, 0].store(m_i[rs]), stats[block_bh, split, slot, 1].store(l_i[rs])
    ).end(rs)
    return (
        UOp.group(out[rtid].store(acc), stat_stores)
        .end(row, wave_n, lane)
        .end(split, block_bh)
        .sink(arg=KernelInfo(name="flash_decode_mma", opts_to_apply=()))
    )


@functools.cache
def _combine(o: UOp, part: UOp, stats: UOp, valid_first: Any, group: int, h_kv: int) -> UOp:
    # one wave per (batch * KV head, row, head of the group, 64-dim tile); every lane weighs the same blocks
    valid_first = _unbind(valid_first)
    _, _, M, D = o.shape
    BKV, splits, _, _ = part.shape
    DT = 64 if D % 64 == 0 else WARP_SIZE
    assert D % DT == 0
    pairs = M * group
    block, block_dt = UOp.range(BKV * pairs, 0, AxisType.GLOBAL), UOp.range(D // DT, 1, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, 2, AxisType.LOCAL)
    bkv, pair = block // pairs, block % pairs
    row, g, b, kv_head = pair // group, pair % group, bkv // h_kv, bkv % h_kv
    slot, head = row * SLOTS + g, kv_head * group + g
    NPD = DT // WARP_SIZE
    dims = tuple(block_dt * DT + lane * NPD + i for i in range(NPD))
    live = ((valid_first + row + BLOCK_N - 1) // BLOCK_N).minimum(splits)  # the blocks that saw this row's keys

    def iloop(ph: UOp, val: float) -> UOp:
        return ph.store(ph.const_like(val))

    s1 = UOp.range(live, 100)
    top = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
    top_i = top.after(iloop(top, -math.inf))
    update0 = top_i.store(top_i.after(s1).maximum(stats[bkv, s1, slot, 0].load())).end(s1)
    top = top_i.after(update0)
    s2 = UOp.range(live, 101)
    acc = UOp.alloc((NPD,), dtypes.float, addrspace=AddrSpace.REG)
    total = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
    acc_i, total_i = acc.after(iloop(acc, 0)), total.after(iloop(total, 0))
    w = ((stats[bkv, s2, slot, 0].load() - top) * LOG2E).exp2()
    update1 = UOp.group(
        *[acc_i[i].store(acc_i.after(s2)[i].load() + w * part[bkv, s2, slot, d].load()) for i, d in enumerate(dims)],
        total_i[0].store(total_i.after(s2)[0].load() + w * stats[bkv, s2, slot, 1].load()),
    ).end(s2)
    acc, total = acc_i.after(update1), total_i.after(update1)
    inv = 1 / total[0].load()
    return (
        UOp.group(*[o[b, head, row, d].store(acc[i].load() * inv) for i, d in enumerate(dims)])
        .end(lane, block_dt, block)
        .sink(arg=KernelInfo(name="flash_decode_mma_combine", opts_to_apply=()))
    )


def supported(q: Tensor, cache_kv: Tensor) -> bool:
    """Shapes and devices this kernel serves: gfx11 (gfx12's fragment layout is written but untested), a head size
    the matrix tiles divide, at most 16 query heads per KV head, a cache whose length the key tiles divide."""
    if not amd_custom_kernels_supported(q.device) or _is_rdna4(q.device) or _is_cdna(q.device):
        return False
    D, N = int(q.shape[3]), int(cache_kv.shape[3])
    group = int(q.shape[1]) // int(cache_kv.shape[2])
    return bool(D % (WAVES_N * LANES_PER_WAVE_N) == 0 and D % WMMA_K == 0 and N % BLOCK_N == 0 and group <= SLOTS)


def installed() -> bool:
    """Whether decoding runs on this kernel in this process (`tinyllm.tg.kernels.install` with "attention")."""
    from tinygrad.llm import model

    return hasattr(model.flash_attention, "_tinyllm_original")


def decode_attention(original: Any) -> Any:
    """`tinygrad.llm.kernels.amd.flash_attention` with one-token calls on this kernel: what decoding must use for
    the speculative verify to equal it (`install`)."""

    def flash_attention(q: Tensor, assigned_kv: Tensor, valid_end: Any) -> Tensor:
        if resolve(q.shape[2] == 1, False) and supported(q, assigned_kv):
            # realized on its own, as the verify realizes each row's query (norm and RoPE round alike)
            return flash_decode_mma(q.half().contiguous(), assigned_kv, valid_end)
        return original(q, assigned_kv, valid_end)

    flash_attention._tinyllm_original = original  # type: ignore[attr-defined]
    return flash_attention


def flash_decode_mma(q: Tensor, cache_kv: Tensor, valid_first: Any) -> Tensor:
    """Attention of `q` (B, H, M, D), row j over the first `valid_first + j` positions of `cache_kv`
    (2, B, H_KV, N, D) half: (B, H, M, D) float32. Each row's result is the same for any M."""
    M = int(q.shape[2])
    if M <= MAX_ROWS and M & (M - 1) == 0:
        return _launch(q, cache_kv, valid_first)
    outs, start = [], 0
    while start < M:  # launches of a power of two of rows each: a row's work does not depend on the others
        n = 1 << (min(M - start, MAX_ROWS).bit_length() - 1)
        outs.append(_launch(q[:, :, start : start + n], cache_kv, valid_first + start))
        start += n
    return outs[0].cat(*outs[1:], dim=2)


def _launch(q: Tensor, cache_kv: Tensor, valid_first: Any) -> Tensor:
    if isinstance(valid_first, UOp):  # carry the length binding into the graph, as decoding does
        cache_kv = Tensor(cache_kv.uop.after(valid_first))
    B, H, M, D = q.shape
    H_KV = int(cache_kv.shape[2])
    group = H // H_KV
    # (B * KV heads, M * SLOTS, D): slot j * SLOTS + g is row j, head g of its KV head; the padding slots are zero
    qp = q.half().reshape(B, H_KV, group, M, D).permute(0, 1, 3, 2, 4)
    qp = qp.pad((None, None, None, (0, SLOTS - group), None)).reshape(B * H_KV, M * SLOTS, D).contiguous()
    axis = q.uop.axis
    part = Tensor.empty(B * H_KV, SPLITS, M * SLOTS, D, dtype="float32", device=q.device, axis=axis)
    stats = Tensor.empty(B * H_KV, SPLITS, M * SLOTS, 2, dtype="float32", device=q.device, axis=axis)
    fxn = functools.partial(_partial, valid_first=valid_first, rows=M, rdna4=_is_rdna4(q.device))
    part, stats = Tensor.custom_kernel(part, stats, qp, cache_kv, fxn=fxn)[:2]
    out = Tensor.empty(B, H, M, D, dtype="float32", device=q.device, axis=axis)
    fxn = functools.partial(_combine, valid_first=valid_first, group=group, h_kv=H_KV)
    return Tensor.custom_kernel(out, part, stats, fxn=fxn)[0]
