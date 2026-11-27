"""Qwen3.8-Flash-Next's sparse attention (QSA) on gfx11: each query reads at most 2,051 positions, whatever the
context, instead of the whole history.

A QSA layer's indexer scores blocks of 4 consecutive positions: a block's key is the mean of its 4 raw indexer
keys, RMS-normalised and rotated at the block's first position; its score for a query is the sum over the 4
indexer heads of relu(q_h . key) / sqrt(128). The query attends to its 512 best complete blocks and to its
incomplete block (the tail: positions from the last multiple of 4 up to itself). While it sees at most 512
complete blocks it takes them all, and the layer is dense causal attention.

tinygrad's path (`tinygrad.llm.model.Indexer`, `TransformerBlock._attention`) builds a mask over the whole cache
and runs its generic attention through it. At 16,000 positions that was 138 of a 196 ms decode step: about 11.5 ms
a layer, growing with every token (docs/04-qwen3.8-flash-next.md, section 12). It also pools, normalises and
rotates every block key of the configured context again on every call, and sorts all the blocks. Here, as in
Radiance's design (section 6), four kernels do it:

  block keys  the keys of the blocks this call's tokens fall in, computed once into a cache of block keys
              (float32, dimension-major so the scoring reads them coalesced). A block's key is recomputed while
              it fills, so it is final when its fourth token arrives; it is indexed by position and needs no
              checkpoint, like the raw keys it comes from.
  scores      one thread per (query, complete block) of the live history, not of the configured context.
  select      one workgroup per query: an exact top 512 in linear time. A 16-way search over the float bits finds
              the 512th largest score in 8 passes (scores are sums of relus, so never negative, and their bits order
              like their values); then every thread counts its contiguous run of blocks above it and equal to it, a
              prefix sum gives each thread where its blocks go, and they are written in ascending order. Equal
              scores at the threshold are taken lowest block first. A query that sees at most 512 blocks takes
              them all without reading a score. The list ends with the tail block.
  attention   paged attention over the list, 8 list entries (32 keys) per tile, on the matrix cores. It is
              `tinyllm.tg.kernels.decode_mma`'s tile loop with one query row per workgroup and the row a grid axis:
              each row gathers its own keys. Positions past the query (in its tail block) and entries past its list
              are masked, and their values are zeroed when loaded. A second kernel combines the splits.
  grouped     for prefill calls of GROUP_MIN_ROWS rows or more: consecutive rows' lists overlap, so groups of
              GROUP rows share their keys. One kernel merges a group's lists into their union with, per entry,
              which rows listed it (a mark per block in local memory, then a prefix-sum compaction); another runs
              the group's GROUP x 12 queries (12 query heads per KV head, three matrix tiles, no padding) over the
              union's tiles, each gathered once, with a key counting for a query when its row listed the block.
              On 16,384 tokens of code (`UD-Q4_K_XL`, 2,048-token calls) prefill went from 773 to 797 tok/s. A
              row's result depends on its group, which depends only on the call's own rows, so a request resumed
              from a checkpoint repeats the same calls and the same groups.

Selected blocks keep the scores' float32 rounding, so a near tie can resolve differently from tinygrad's sort
(and llama.cpp's); the selection itself is exact for the scores computed here.

Mirrors `TransformerBlock._attention` and `Indexer.__call__` at the pinned commit for what it replaces; re-check
on every tinygrad bump.
"""

from __future__ import annotations

import functools
import math
from typing import Any

from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.llm import model as llm_model
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
    _amd_load,
    _is_cdna,
    _is_rdna4,
    _reg,
    _unbind,
    _wmma_fragment,
    amd_custom_kernels_supported,
    warp_reduce,
)
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

THREADS = 256  # threads of a scoring or selection workgroup
SLOTS = WMMA_M  # a row's query heads of one KV head, padded to a matrix tile
WAVES_N = 4  # waves per attention workgroup: they share out the output dimensions
SPLITS_DECODE = 32  # workgroups a decoded row's tiles are dealt to, per KV head (at most 65 tiles)
SPLITS_PREFILL = 4  # the same for prefill rows, which fill the machine by themselves
SPLITS_LARGE, LARGE_ROWS = 1, 512  # from this many rows one workgroup takes all of a row's tiles: no partials
GROUPED = True  # prefill rows in groups sharing their keys (`_union`, `_attend_group`); off: one row a workgroup
GROUP, WAVES_D = 4, 2  # rows of a group; waves sharing out the output dimensions of the group attention
GROUP_MIN_ROWS = 64  # fewer rows (decode, the speculative verify) keep the one-row kernel
MARKS_BYTES = 60 * 1024  # the union's local memory: one int32 per block of the cache
GRID = {"head": 0, "split": 1, "row": 2}  # range ids of the attention kernel's grid axes
NO_SCORE = 0x7F800001  # the bits after +inf: no score is at least this
WAYS = 16  # intervals each pass of the threshold search splits its interval into
SEARCH_PASSES = 8  # passes that close the 2^31 span of the bits, 16 times narrower each
_enabled = False


def _ireg(shape: tuple[int, ...], value: int, dep: UOp | None = None) -> UOp:
    ret = UOp.alloc(shape, dtypes.int32, addrspace=AddrSpace.REG)
    return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))


def _int(x: int) -> UOp:
    return UOp.const(x, dtypes.int32)


def _list_len(q_pos: UOp, kpool: int, topb: int) -> UOp:
    """Entries of a query's list: its selected complete blocks, then its tail block if it has one."""
    return ((q_pos + 1) // kpool).minimum(topb) + ((q_pos + 1) % kpool).minimum(1)


@functools.cache
def _block_keys(
    blk: UOp, kg: UOp, weight: UOp, freqs: UOp, q_first: Any, rows: Any, most: int, eps: float, kpool: int
) -> UOp:
    """Keys of the blocks positions q_first .. q_first + rows - 1 fall in (at most `most`): mean of the raw keys,
    RMSNorm, rotary at the block's start."""
    q_first, rows = _unbind(q_first), _unbind(rows)
    B, D, _ = blk.shape  # (batch, 128, blocks) float32
    rope = freqs.shape[1]  # cos for rope / 2 dimensions, then sin
    assert D == 4 * WARP_SIZE and rope == 2 * WARP_SIZE, "one wave per block: 4 dimensions a lane, rotary on 64"
    bb, nth = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(most, 1, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, 2, AxisType.LOCAL)
    first = q_first // kpool
    inside = first + nth <= (q_first + rows - 1) // kpool
    block = inside.where(first + nth, first)
    dims = [lane + i * WARP_SIZE for i in range(4)]  # rotary pairs dimension d with d + 32
    xs = []
    for d in dims:
        total = kg[bb, block * kpool, d].load().cast(dtypes.float32)
        for j in range(1, kpool):
            total = total + kg[bb, block * kpool + j, d].load().cast(dtypes.float32)
        xs.append(total / kpool)
    sq = warp_reduce(sum((x * x for x in xs[1:]), xs[0] * xs[0]), full_wave=True)
    inv = (sq / D + eps).rsqrt()
    ys = [x * inv * weight[d].load() for x, d in zip(xs, dims, strict=True)]
    cos, sin = freqs[block * kpool, lane].load(), freqs[block * kpool, lane + WARP_SIZE].load()
    out = [ys[0] * cos - ys[1] * sin, ys[1] * cos + ys[0] * sin, ys[2], ys[3]]
    stores = [blk[bb, d, block.valid(inside)].store(o) for o, d in zip(out, dims, strict=True)]
    return UOp.group(*stores).end(lane, nth, bb).sink(arg=KernelInfo(name="qsa_block_keys", opts_to_apply=()))


@functools.cache
def _scores(out: UOp, iq: UOp, blk: UOp, q_first: Any, rows: Any, kpool: int, topb: int) -> UOp:
    """Score of every complete block of the last row's history, for each valid row."""
    q_first, rows = _unbind(q_first), _unbind(rows)
    B, _, H, D = iq.shape  # (batch, rows, heads, 128) float32
    n_last = (q_first + rows) // kpool  # complete blocks the last row sees
    chunks = (n_last + THREADS - 1) // THREADS
    bb, row = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(rows, 1, AxisType.GLOBAL)
    chunk, tid = UOp.range(chunks, 2, AxisType.GLOBAL), UOp.range(THREADS, 3, AxisType.LOCAL)
    block = chunk * THREADS + tid
    inside = block < n_last
    safe = inside.where(block, 0)
    keys = [blk[bb, d, safe].load() for d in range(D)]
    total = UOp.const(0, dtypes.float32)
    for h in range(H):
        dot = iq[bb, row, h, 0].load() * keys[0]
        for d in range(1, D):
            dot = dot + iq[bb, row, h, d].load() * keys[d]
        total = total + dot.maximum(0) * D**-0.5
    store = out[bb, row, block.valid(inside)].store(total)
    return store.end(tid, chunk, row, bb).sink(arg=KernelInfo(name="qsa_scores", opts_to_apply=()))


@functools.cache
def _select(sel: UOp, scores: UOp, q_first: Any, rows: Any, kpool: int, topb: int) -> UOp:
    """Each row's list: its best `topb` complete blocks in ascending order (all of them if it has no more), then
    its tail block. Rows past `rows` (padding) repeat the last valid row."""
    q_first, rows = _unbind(q_first), _unbind(rows)
    B, R, _ = sel.shape
    waves = THREADS // WARP_SIZE
    bb, row = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(R, 1, AxisType.GLOBAL)
    tid = UOp.range(THREADS, 2, AxisType.LOCAL)
    wave = tid // WARP_SIZE
    srow = row.minimum(rows - 1)
    n = (q_first + srow + 1) // kpool  # this row's complete blocks
    dense = n <= topb
    one, zero = _int(1), _int(0)

    # the largest threshold that at least topb scores reach, as float bits: lo reaches it, hi does not. Each pass
    # counts the scores reaching WAYS - 1 thresholds evenly spaced between them and keeps the interval where the
    # count drops below topb, so the bits' 2^31 span closes in SEARCH_PASSES passes over the scores
    lohi = UOp.alloc((2,), dtypes.int32, addrspace=AddrSpace.REG)
    lohi = lohi.after(lohi[0].store(zero), lohi[1].store(_int(NO_SCORE)))
    it = UOp.range(dense.where(0, SEARCH_PASSES), 100)
    cur = lohi.after(it)
    lo, hi = cur[0].load(), cur[1].load()
    stride = (hi - lo + (WAYS - 1)) // WAYS
    marks = [(lo + stride * i).minimum(hi) for i in range(1, WAYS)]
    thresholds = [m.bitcast(dtypes.float32) for m in marks]
    cnt = _ireg((WAYS - 1,), 0, it)
    j = UOp.range((n + THREADS - 1) // THREADS, 101)
    e = j * THREADS + tid
    ok = e < n
    value = scores[bb, srow, ok.where(e, 0)].load()
    tally = cnt.after(j)
    cnt = cnt.after(
        UOp.group(
            *[cnt[i].store(tally[i].load() + (ok & (value >= t)).where(one, zero)) for i, t in enumerate(thresholds)]
        ).end(j)
    )
    wave_sums = UOp.alloc((waves, WAYS - 1), dtypes.int32, addrspace=AddrSpace.LOCAL)
    wave_sums = wave_sums.after(
        *[wave_sums[wave, i].store(warp_reduce(cnt[i].load(), Ops.ADD, full_wave=True)) for i in range(WAYS - 1)]
    )
    totals = [sum((wave_sums[w, i].load() for w in range(1, waves)), wave_sums[0, i].load()) for i in range(WAYS - 1)]
    # the counts fall as the thresholds rise: the marks with enough scores are the first `reached`
    reached = sum(((t >= topb).where(one, zero) for t in totals[1:]), (totals[0] >= topb).where(one, zero))
    new_lo = functools.reduce(lambda acc, im: reached.eq(im[0]).where(im[1], acc), enumerate(marks, 1), lo)
    new_hi = functools.reduce(lambda acc, im: reached.eq(im[0]).where(im[1], acc), enumerate(marks), hi)
    step = UOp.group(lohi[0].store(new_lo), lohi[1].store(new_hi)).end(it)
    tau = dense.where(UOp.const(-1.0, dtypes.float32), lohi.after(step)[0].load().bitcast(dtypes.float32))

    # each thread's contiguous run of blocks: how many are above the threshold and how many equal it
    run = (n + THREADS - 1) // THREADS
    start = tid * run

    def classify(a: UOp) -> tuple[UOp, UOp, UOp]:
        idx = start + a
        inside = idx < n
        value = scores[bb, srow, inside.where(idx, 0)].load()
        above = inside & (dense | (value > tau))
        return idx, above, inside & dense.logical_not() & value.eq(tau)

    counts = UOp.alloc((2,), dtypes.int32, addrspace=AddrSpace.REG)
    counts = counts.after(counts[0].store(zero), counts[1].store(zero))
    a = UOp.range(run, 102)
    _, above, equal = classify(a)
    tally = counts.after(a)
    counts = counts.after(
        UOp.group(
            counts[0].store(tally[0].load() + above.where(one, zero)),
            counts[1].store(tally[1].load() + equal.where(one, zero)),
        ).end(a)
    )
    mine_above, mine_equal = counts[0].load(), counts[1].load()
    # inclusive prefix sums over the threads (Hillis-Steele, double-buffered in local memory)
    scan = UOp.alloc((2, 2, THREADS), dtypes.int32, addrspace=AddrSpace.LOCAL)
    buf = scan.after(scan[0, 0, tid].store(mine_above), scan[0, 1, tid].store(mine_equal))
    p, off = 0, 1
    while off < THREADS:
        prev = (tid - off).maximum(0)
        has = tid >= off
        g = buf[p, 0, tid].load() + has.where(buf[p, 0, prev].load(), zero)
        h = buf[p, 1, tid].load() + has.where(buf[p, 1, prev].load(), zero)
        buf = scan.after(scan[1 - p, 0, tid].store(g), scan[1 - p, 1, tid].store(h))
        p, off = 1 - p, off * 2
    before_above = buf[p, 0, tid].load() - mine_above
    before_equal = buf[p, 1, tid].load() - mine_equal
    need = topb - buf[p, 0, THREADS - 1].load()  # equal scores to take, lowest blocks first
    base = before_above + before_equal.minimum(need)

    written = UOp.alloc((2,), dtypes.int32, addrspace=AddrSpace.REG)
    written = written.after(written[0].store(zero), written[1].store(zero))
    a2 = UOp.range(run, 103)
    idx, above, equal = classify(a2)
    state = written.after(a2)
    taken, seen = state[0].load(), state[1].load()
    take = above | (equal & (before_equal + seen < need))
    slot = (base + taken).cast(dtypes.weakint)
    write = UOp.group(
        sel[bb, row, slot.valid(take)].store(idx.cast(dtypes.int32)),
        written[0].store(taken + take.where(one, zero)),
        written[1].store(seen + equal.where(one, zero)),
    ).end(a2)
    tail = sel[bb, row, n.minimum(topb).valid(tid.eq(0))].store(n.cast(dtypes.int32))
    return UOp.group(write, tail).end(tid, row, bb).sink(arg=KernelInfo(name="qsa_select", opts_to_apply=()))


@functools.cache
def _union(
    ulist: UOp, umask: UOp, ulen: UOp, sel: UOp, q_first: Any, rows: Any, kpool: int, topb: int, nblk: int
) -> UOp:
    """For each group of GROUP consecutive rows: the union of their lists, ascending (`ulist`, `ulen` entries), and
    for each entry the rows that listed it (`umask`, bit i for row i of the group). Padding rows repeat the last
    valid row's list (`_select`), so they list what it lists. `nblk` bounds the blocks a list can name."""
    q_first, rows = _unbind(q_first), _unbind(rows)
    B, R, _ = sel.shape
    bb, grp = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(R // GROUP, 1, AxisType.GLOBAL)
    tid = UOp.range(THREADS, 2, AxisType.LOCAL)
    one, zero = _int(1), _int(0)
    last_pos = q_first + (grp * GROUP + GROUP - 1).minimum(rows - 1)
    nb = (last_pos + 1 + kpool - 1) // kpool  # blocks the group's lists can name: 0 .. nb - 1
    marks = UOp.alloc((nblk,), dtypes.int32, addrspace=AddrSpace.LOCAL)
    zj = UOp.range((nb + THREADS - 1) // THREADS, 100)
    zi = zj * THREADS + tid
    marks = marks.after(marks[zi.valid(zi < nb)].store(zero).end(zj))
    per = (topb + 1 + THREADS - 1) // THREADS
    for i in range(GROUP):
        row = grp * GROUP + i
        n_list = _list_len(q_first + row.minimum(rows - 1), kpool, topb)
        stores = []
        for j in range(per):
            e = tid + j * THREADS
            ok = e < n_list
            blk = sel[bb, row, ok.where(e, 0)].load().cast(dtypes.weakint).maximum(0).minimum(nblk - 1)
            stores.append(marks[blk.valid(ok)].store(marks[blk].load() | (1 << i)))
        marks = marks.after(*stores)  # a row's blocks are distinct: no two threads of one pass share an entry
    run = (nb + THREADS - 1) // THREADS
    start = tid * run
    cnt = _ireg((1,), 0)
    a = UOp.range(run, 101)
    idx = start + a
    inside = idx < nb
    present = inside & marks[inside.where(idx, 0)].load().ne(0)
    cnt = cnt.after(cnt[0].store(cnt.after(a)[0].load() + present.where(one, zero)).end(a))
    mine = cnt[0].load()
    scan = UOp.alloc((2, THREADS), dtypes.int32, addrspace=AddrSpace.LOCAL)
    buf = scan.after(scan[0, tid].store(mine))
    p, off = 0, 1
    while off < THREADS:
        prev, has = (tid - off).maximum(0), tid >= off
        buf = scan.after(scan[1 - p, tid].store(buf[p, tid].load() + has.where(buf[p, prev].load(), zero)))
        p, off = 1 - p, off * 2
    base = buf[p, tid].load() - mine
    written = _ireg((1,), 0)
    a2 = UOp.range(run, 102)
    idx2 = start + a2
    inside2 = idx2 < nb
    mask = marks[inside2.where(idx2, 0)].load()
    take = inside2 & mask.ne(0)
    taken = written.after(a2)[0].load()
    slot = (base + taken).cast(dtypes.weakint).valid(take)
    write = UOp.group(
        ulist[bb, grp, slot].store(idx2.cast(dtypes.int32)),
        umask[bb, grp, slot].store(mask),
        written[0].store(taken + take.where(one, zero)),
    ).end(a2)
    total = ulen[bb, grp, UOp.const(0, dtypes.weakint).valid(tid.eq(THREADS - 1))].store(buf[p, tid].load())
    return UOp.group(write, total).end(tid, grp, bb).sink(arg=KernelInfo(name="qsa_union", opts_to_apply=()))


@functools.cache
def _attend_group(
    o: UOp, q: UOp, cache: UOp, ulist: UOp, umask: UOp, ulen: UOp, q_first: Any, rows: Any, kpool: int, rdna4: bool
) -> UOp:
    """Attention of a group of GROUP rows over the union of their lists (`_union`), one workgroup per (KV head,
    group): each union tile of BLOCK_N keys is gathered once for the group's GROUP x (query heads of the KV head)
    queries, packed into matrix tiles. A key counts for a query when the query's row listed its block and it is not
    past the query. Waves split the queries (one matrix tile each) and the output dimensions; the result is
    normalised here (one workgroup sees all of a group's keys)."""
    q_first, rows = _unbind(q_first), _unbind(rows)
    BKV, RQ, D = q.shape  # (batch * KV heads, rows * heads per KV head, head size) half
    _, B, H_KV, N, cache_dim = cache.shape
    _, H, R, _ = o.shape
    HG = H // H_KV
    QM = GROUP * HG
    WM = QM // WMMA_M  # waves along the queries
    assert QM % WMMA_M == 0 and R % GROUP == 0 and RQ == R * HG and cache_dim == D and D % (8 * WAVES_D) == 0
    k, v = cache[0].reshape(B * H_KV, N * D), cache[1].reshape(B * H_KV, N * D)
    threads = WARP_SIZE * WM * WAVES_D
    TM, TN = WMMA_M // LANES_PER_WAVE_M, BLOCK_N // LANES_PER_WAVE_N
    TD, SCALE = D // (WAVES_D * LANES_PER_WAVE_N), 1 / math.sqrt(D)
    entries, maxu = BLOCK_N // kpool, int(ulist.shape[2])
    Q_ELEMS, CHUNKS = QM * D // threads, BLOCK_N * D // 8  # Q values per thread; 8-value chunks of a K or V tile
    assert TN == WAVES_D, "each wave of a pair scores one 16-key matrix tile of a key tile"
    assert QM * D % threads == 0 and D % Q_ELEMS == 0
    block_bh, grp = UOp.range(BKV, 0, AxisType.GLOBAL), UOp.range(R // GROUP, 1, AxisType.GLOBAL)
    wave_m, wave_n = UOp.range(WM, 2, AxisType.LOCAL), UOp.range(WAVES_D, 3, AxisType.LOCAL)
    lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
    tid = (wave_m * WAVES_D + wave_n) * WARP_SIZE + lane
    lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
    b, kv_head = block_bh // H_KV, block_bh % H_KV
    n_u = ulen[b, grp, 0].load().cast(dtypes.weakint).maximum(0).minimum(maxu)
    last_pos = q_first + (grp * GROUP + GROUP - 1).minimum(rows - 1)  # keys past it are not written yet
    Q_lds = UOp.alloc((QM, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
    q_rows = q.reshape(BKV, R // GROUP, QM * D)[block_bh, grp].reshape(threads, Q_ELEMS)
    qv = Q_lds.after(Q_lds.reshape(threads, Q_ELEMS)[tid].store(q_rows[tid]))
    # one buffer for the K tile (keys x dims), then the V tile (dims x keys): QK is done before V is copied
    KV = UOp.alloc((max(BLOCK_N * (D + LDS_PAD), D * (BLOCK_N + LDS_PAD)),), dtypes.half, addrspace=AddrSpace.LOCAL)
    K_lds = KV[: BLOCK_N * (D + LDS_PAD)].reshape(BLOCK_N, D + LDS_PAD)[:, :D]
    V_lds = KV[: D * (BLOCK_N + LDS_PAD)].reshape(D, BLOCK_N + LDS_PAD)[:, :BLOCK_N]
    acc, m_i, l_i = _reg((TM, TD), 0), _reg((TM,), -math.inf), _reg((TM,), 0)
    tile = UOp.range((n_u + entries - 1) // entries, 100)

    def entry_of(kidx: UOp) -> tuple[UOp, UOp, UOp]:
        """(listed, union entry, position) of key kidx of this tile."""
        entry = tile * entries + kidx // kpool
        listed = entry < n_u
        safe = listed.where(entry, 0)
        pos = ulist[b, grp, safe].load().cast(dtypes.weakint) * kpool + kidx % kpool
        return listed, safe, pos

    def copy_tile(target: Any, src: UOp, zero_unwritten: bool) -> list[UOp]:
        stores = []
        for c in range(-(-CHUNKS // threads)):
            chunk = tid + c * threads
            ok = chunk < CHUNKS
            key, d0 = (chunk // (D // 8)).minimum(BLOCK_N - 1), (chunk % (D // 8)) * 8
            listed, _, pos = entry_of(key)
            live = listed & (pos <= last_pos)
            vec = _amd_load(src[block_bh, live.where(pos, 0) * D + d0], 8)
            for e in range(8):
                value = live.where(vec[e], vec[e].const_like(0)) if zero_unwritten else vec[e]
                stores.append(target[key.valid(ok) if CHUNKS % threads else key, d0 + e].store(value))
        return stores

    K_tile = K_lds.after(*copy_tile(K_lds.after(tile), k, False))
    # each wave of a pair scores its own 16 of the tile's keys; the pair splits the output dimensions for PV
    S_reg = _reg((TM, 1), 0, tile)
    k_qk, tm1 = UOp.range(D // WMMA_K, 101), UOp.range(TM // WMMA_ACC, 200)
    S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, 1).permute(0, 2, 1)[tm1, 0]
    q_frag = qv.after(tile).reshape(WM, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
    k_frag = K_tile.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[wave_n, lane_n, k_qk]
    q_frag, k_frag = (_wmma_fragment(f, lane, rdna4) for f in (q_frag, k_frag))
    qk = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1).end(k_qk).barrier()
    S_reg = S_reg.after(qk, S_reg.store(S_reg * SCALE))

    def query_index(rm: UOp | int) -> UOp:
        """This lane's query rm, as an index into the group's queries."""
        return wave_m * WMMA_M + (lane_m * TM + rm if rdna4 else rm * LANES_PER_WAVE_M + lane_m)

    def query(rm: UOp | int) -> tuple[UOp, UOp]:
        """(row of the group, query head) of this lane's query rm."""
        ql = query_index(rm)
        return ql // HG, ql % HG

    rm = UOp.range(TM, 250)
    r_in, _ = query(rm)
    q_pos = q_first + (grp * GROUP + r_in).minimum(rows - 1)
    listed, entry, pos = entry_of(wave_n * LANES_PER_WAVE_N + lane_n)
    member = ((umask[b, grp, entry].load() >> r_in.cast(dtypes.int32)) & 1).eq(1)
    attended = listed & member & (pos <= q_pos)
    masked = attended.where(S_reg[rm, 0], S_reg[rm, 0].const_like(-math.inf))
    S_reg = S_reg.after(S_reg[rm, 0].store(masked).end(rm))
    # the row maxima over this wave's keys, then over the pair's, exchanged through local memory
    m_part, ri_w = _reg((TM,), -math.inf, tile), UOp.range(TM, 270)
    m_part = m_part.after(m_part[ri_w].store(warp_reduce(S_reg[ri_w, 0], Ops.MAX)).end(ri_w))
    M_lds = UOp.alloc((WAVES_D, QM), dtypes.float, addrspace=AddrSpace.LOCAL)
    rw = UOp.range(TM, 272)
    M_tile = M_lds.after(M_lds[wave_n, query_index(rw)].store(m_part[rw]).end(rw))
    m_ij, rx = _reg((TM,), -math.inf, tile), UOp.range(TM, 274)
    pair_max = M_tile[0, query_index(rx)].load()
    for w in range(1, WAVES_D):
        pair_max = pair_max.maximum(M_tile[w, query_index(rx)].load())
    m_ij = m_ij.after(m_ij[rx].store(pair_max).end(rx))
    tile_max = m_ij.reshape(TM, 1).maximum(-1e30)
    S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
    p_part, ri_ws = _reg((TM,), 0, tile), UOp.range(TM, 295)
    p_part = p_part.after(p_part[ri_ws].store(warp_reduce(S_reg[ri_ws, 0])).end(ri_ws))
    # one probability tile for the pair: this wave writes its keys' columns, and its rows' partial sums
    P_lds = UOp.alloc((QM, BLOCK_N), dtypes.half, addrspace=AddrSpace.LOCAL)
    L_lds = UOp.alloc((WAVES_D, QM), dtypes.float, addrspace=AddrSpace.LOCAL)
    rp, rl = UOp.range(TM, 296), UOp.range(TM, 297)
    key = wave_n * LANES_PER_WAVE_N + lane_n
    P_store = P_lds[query_index(rp), key].store(S_reg[rp, 0].cast(dtypes.half)).end(rp)
    L_store = L_lds[wave_n, query_index(rl)].store(p_part[rl]).end(rl)
    # values of keys not yet written are zeroed: garbage times a zero probability need not be zero
    V_store = copy_tile(V_lds.after(qk).permute(1, 0), v, True)
    P_tile = P_lds.after(P_store, L_store, *V_store)
    V_tile = V_lds.after(P_store, L_store, *V_store)
    L_tile = L_lds.after(P_store, L_store, *V_store)
    beta_i, ri4, rj4 = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG), UOp.range(TM, 330), UOp.range(TD, 331)
    p_sum = L_tile[0, query_index(ri4)].load()
    for w in range(1, WAVES_D):
        p_sum = p_sum + L_tile[w, query_index(ri4)].load()
    # a union tile can hold no key of a row: with no key seen yet both maxima are -inf, and -inf - -inf would
    # make the rescaling NaN; clamped, the rescale and the tile's weight are both 0
    m_new = m_i[ri4].maximum(m_ij[ri4]).maximum(-1e30)
    alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
    correction = UOp.group(
        acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
        l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum),
        m_i[ri4].store(m_new),
        beta_i[ri4].store(beta_val),
    ).end(ri4)
    acc, l_i, m_i, beta_i = (
        acc.after(correction),
        l_i.after(correction),
        m_i.after(correction),
        beta_i.after(correction),
    )
    pv_acc = _reg((TM, TD), 0, tile)
    k_pv, tm2, tn2 = UOp.range(BLOCK_N // WMMA_K, 400), UOp.range(TM // WMMA_ACC, 401), UOp.range(TD, 402)
    pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
    p_frag = P_tile.reshape(WM, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
    v_frag = V_tile.reshape(WAVES_D, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
    p_frag, v_frag = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_frag))
    pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
    pv_acc = pv_acc.after(pv_done)
    ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
    tile_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).end(tile)
    acc, l_i = acc.after(tile_end), l_i.after(tile_end)
    stores = []
    for rm_ in range(TM):
        r_in_, head = query(rm_)
        inv = 1 / l_i[rm_].load()
        for td in range(TD):
            dim = (wave_n * TD + td) * WMMA_N + lane_n
            stores.append(o[b, kv_head * HG + head, grp * GROUP + r_in_, dim].store(acc[rm_, td].load() * inv))
    flops = 4 * BKV * R * HG * maxu * kpool * D  # the most a launch can do: every union entry for every query
    estimates = Estimates(ops=flops, lds=flops // 8, mem=BKV * R * HG * D * 6)
    info = KernelInfo(name="qsa_attention_group", opts_to_apply=(), estimates=estimates, optimized=True)
    return UOp.group(*stores).end(wave_m, wave_n, lane).end(grp, block_bh).sink(arg=info)


@functools.cache
def _attend_partial(
    part: UOp, stats: UOp, q: UOp, cache: UOp, sel: UOp, q_first: Any, rows: Any, kpool: int, topb: int, rdna4: bool
) -> UOp:
    q_first, rows = _unbind(q_first), _unbind(rows)
    BKV, RS, D = q.shape  # (batch * KV heads, rows * SLOTS, head size) half
    _, B, H_KV, N, cache_dim = cache.shape
    R, splits, entries = RS // SLOTS, part.shape[1], BLOCK_N // kpool
    k, v = cache[0].reshape(B * H_KV, N, cache_dim), cache[1].reshape(B * H_KV, N, cache_dim)
    threads = WARP_SIZE * WAVES_N
    TM, TN = SLOTS // LANES_PER_WAVE_M, BLOCK_N // LANES_PER_WAVE_N
    TD, SCALE = D // (WAVES_N * LANES_PER_WAVE_N), 1 / math.sqrt(D)
    KV_ELEMS, Q_ELEMS = BLOCK_N * D // threads, SLOTS * D // threads
    assert D % WMMA_K == 0 and D % KV_ELEMS == 0 and BLOCK_N % kpool == 0
    longest = (_list_len(q_first + rows - 1, kpool, topb) + 1 + entries - 1) // entries  # tiles of the longest list
    block_bh = UOp.range(BKV, GRID["head"], AxisType.GLOBAL)
    split = UOp.range(longest.minimum(splits), GRID["split"], AxisType.GLOBAL)
    row = UOp.range(R, GRID["row"], AxisType.GLOBAL)
    wave_n, lane = UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
    tid = wave_n * WARP_SIZE + lane
    lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
    b = block_bh // H_KV
    q_pos = q_first + row.minimum(rows - 1)
    n_list = _list_len(q_pos, kpool, topb)
    tiles = (row < rows).where((n_list + entries - 1) // entries, 0)  # padding rows do no work (sliced off)
    # the row's query slots, copied to local memory once
    Q_lds = UOp.alloc((SLOTS, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
    q_rows = q.reshape(BKV, R, SLOTS * D)[block_bh, row].reshape(threads, Q_ELEMS)
    qv = Q_lds.after(Q_lds.reshape(threads, Q_ELEMS)[tid].store(q_rows[tid]))
    k, v = k[block_bh], v[block_bh]
    KV_lds = UOp.alloc((BLOCK_N, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
    acc, m_i, l_i = _reg((TM, TD), 0), _reg((TM,), -math.inf), _reg((TM,), 0)
    rnd = UOp.range((tiles - split + splits - 1) // splits, 100)  # this block's tiles: split, split + splits, ...
    tile = split + rnd * splits

    def key(kidx: UOp) -> tuple[UOp, UOp]:
        """Whether key kidx of this tile is one the row attends to, and its position (0 if not)."""
        entry = tile * entries + kidx // kpool
        listed = entry < n_list
        pos = sel[b, row, listed.where(entry, 0)].cast(dtypes.weakint) * kpool + kidx % kpool
        attended = listed & (pos <= q_pos)
        return attended, attended.where(pos, 0)

    per_key = D // KV_ELEMS  # threads loading one key
    mine, mine_pos = key(tid // per_key)
    dim0 = (tid % per_key) * KV_ELEMS
    load_k = UOp.range(KV_ELEMS, 90)
    kval = k.reshape(N * D)[mine_pos * D + dim0 + load_k]
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
    attended, _ = key(rn * LANES_PER_WAVE_N + lane_n)
    masked = attended.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))
    S_reg = S_reg.after(S_reg[rm, rn].store(masked).end(rm, rn))
    m_ij, rm2 = _reg((TM,), -math.inf, rnd), UOp.range(TN, 261)
    m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
    ri_w = UOp.range(TM, 270)
    m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], Ops.MAX)).end(ri_w))
    tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
    S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
    p_local, ri_ws = _reg((TM,), 0, rnd), UOp.range(TM, 295)
    p_sum = p_local.after(p_local[ri_ws].store(sum(warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN))).end(ri_ws))
    # one probability tile; the four waves computed the same scores and store the same values
    P_lds = UOp.alloc((SLOTS, BLOCK_N), dtypes.half, addrspace=AddrSpace.LOCAL)
    # gfx11 distributes even/odd rows between half-waves; gfx12 distributes the low/high eight rows.
    row_shape = (LANES_PER_WAVE_M, TM) if rdna4 else (TM, LANES_PER_WAVE_M)
    row_lane, row_elem = (2, 3) if rdna4 else (3, 2)
    P_write = (
        P_lds.reshape(1, 1, *row_shape, TN, LANES_PER_WAVE_N)
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
    # values of keys the row does not attend to are zeroed: the matrix units do not add a zero probability
    # times a value as an exact zero
    vval = mine.where(v.reshape(N * D)[mine_pos * D + dim0 + load_v], 0)
    V_store = V_copy.reshape(threads, KV_ELEMS)[tid, load_v].store(vval).end(load_v)
    P_lds, V_lds = P_lds.after(P_store, V_store), V_lds.after(P_store, V_store)
    pv_acc = _reg((TM, TD), 0, rnd)
    k_pv, tm2, tn2 = UOp.range(BLOCK_N // WMMA_K, 400), UOp.range(TM // WMMA_ACC, 401), UOp.range(TD, 402)
    pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
    p_frag = P_lds.reshape(TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[tm2, lane_n, k_pv]
    v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
    p_frag, v_frag = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_frag))
    pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
    pv_acc = pv_acc.after(pv_done)
    ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
    rnd_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).end(rnd)
    acc, l_i, m_i = acc.after(rnd_end), l_i.after(rnd_end), m_i.after(rnd_end)
    # this block's unnormalized sum, maximum and total for each slot: the combine kernel weighs the blocks
    out = part.reshape(BKV, splits, R, SLOTS, D)[block_bh, split, row]
    out = (
        out.reshape(1, *row_shape, WAVES_N, TD, LANES_PER_WAVE_N)
        .permute(0, 3, row_lane - 1, 5, row_elem - 1, 4)
        .reshape(threads, TM, TD)
    )
    rs = UOp.range(TM, 420)
    slot = row * SLOTS + (lane_m * TM + rs if rdna4 else rs * LANES_PER_WAVE_M + lane_m)
    stat_stores = UOp.group(
        stats[block_bh, split, slot, 0].store(m_i[rs]), stats[block_bh, split, slot, 1].store(l_i[rs])
    ).end(rs)
    return (
        UOp.group(out[tid].store(acc), stat_stores)
        .end(wave_n, lane)
        .end(row, split, block_bh)
        .sink(arg=KernelInfo(name="qsa_attention", opts_to_apply=()))
    )


@functools.cache
def _attend_combine(
    o: UOp, part: UOp, stats: UOp, q_first: Any, rows: Any, group: int, h_kv: int, kpool: int, topb: int
) -> UOp:
    # one wave per (batch * KV head, row, head of the group, 64-dim tile); every lane weighs the same blocks
    q_first, rows = _unbind(q_first), _unbind(rows)
    _, _, R, D = o.shape
    BKV, splits, _, _ = part.shape
    entries = BLOCK_N // kpool
    DT = 64 if D % 64 == 0 else WARP_SIZE
    assert D % DT == 0
    pairs = R * group
    block, block_dt = UOp.range(BKV * pairs, 0, AxisType.GLOBAL), UOp.range(D // DT, 1, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, 2, AxisType.LOCAL)
    bkv, pair = block // pairs, block % pairs
    row, g, b, kv_head = pair // group, pair % group, bkv // h_kv, bkv % h_kv
    slot, head = row * SLOTS + g, kv_head * group + g
    NPD = DT // WARP_SIZE
    dims = tuple(block_dt * DT + lane * NPD + i for i in range(NPD))
    tiles = (_list_len(q_first + row.minimum(rows - 1), kpool, topb) + entries - 1) // entries
    live = (row < rows).where(tiles.minimum(splits), 0)  # the blocks that saw this row's keys; none for padding

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
        .sink(arg=KernelInfo(name="qsa_attention_combine", opts_to_apply=()))
    )


def _bound(t: Tensor, *values: Any) -> Tensor:
    """`t` carrying the bindings of the symbolic values into the graph, as decoding's attention does."""
    symbolic = [v for v in values if isinstance(v, UOp)]
    return Tensor(t.uop.after(*symbolic)) if symbolic else t


def update_block_keys(
    blk: Tensor, kg: Tensor, weight: Tensor, freqs: Tensor, q_first: Any, rows: Any, padded: int, eps: float, kpool: int
) -> Tensor:
    """Recompute the keys of the blocks that positions q_first .. q_first + rows - 1 (rows at most `padded`) fall
    in, into `blk` (batch, key size, blocks) float32 from the raw keys `kg` (batch, positions, key size) half."""
    most = -(-padded // kpool) + 1  # blocks a call of `padded` rows can touch
    fxn = functools.partial(_block_keys, q_first=q_first, rows=rows, most=most, eps=eps, kpool=kpool)
    return Tensor.custom_kernel(blk, _bound(kg, q_first, rows), weight, freqs, fxn=fxn)[0]


def select(iq: Tensor, blk: Tensor, q_first: Any, rows: Any, kpool: int, topb: int) -> Tensor:
    """Lists (batch, padded rows, topb + 1) int32 for the queries `iq` (batch, padded rows, heads, key size)
    float32 at positions q_first .. q_first + rows - 1: selected blocks ascending, then the tail block."""
    B, R = iq.shape[:2]
    scores = Tensor.empty(B, R, blk.shape[2], dtype=dtypes.float32, device=iq.device)
    fxn = functools.partial(_scores, q_first=q_first, rows=rows, kpool=kpool, topb=topb)
    scores = Tensor.custom_kernel(scores, _bound(iq, q_first, rows), blk, fxn=fxn)[0]
    sel = Tensor.empty(B, R, topb + 1, dtype=dtypes.int32, device=iq.device)
    fxn = functools.partial(_select, q_first=q_first, rows=rows, kpool=kpool, topb=topb)
    return Tensor.custom_kernel(sel, _bound(scores, q_first, rows), fxn=fxn)[0]


def attend(q: Tensor, cache_kv: Tensor, sel: Tensor, q_first: Any, rows: Any, kpool: int, topb: int) -> Tensor:
    """Attention of `q` (B, H, padded rows, D) over the positions each row's list names in `cache_kv`
    (2, B, H_KV, N, D) half: (B, H, padded rows, D) float32."""
    B, H, R, D = q.shape
    H_KV = int(cache_kv.shape[2])
    group = H // H_KV
    nblk = int(cache_kv.shape[3]) // kpool + 1
    if (
        GROUPED
        and R >= GROUP_MIN_ROWS
        and R % GROUP == 0
        and (GROUP * group) % WMMA_M == 0
        and nblk * 4 <= MARKS_BYTES
        and not _is_rdna4(q.device)
    ):
        maxu = min(GROUP * (topb + 1), nblk)
        i32 = functools.partial(Tensor.empty, dtype=dtypes.int32, device=q.device)
        fxn = functools.partial(_union, q_first=q_first, rows=rows, kpool=kpool, topb=topb, nblk=nblk)
        ulist, umask, ulen = Tensor.custom_kernel(
            i32(B, R // GROUP, maxu),
            i32(B, R // GROUP, maxu),
            i32(B, R // GROUP, 1),
            _bound(sel, q_first, rows),
            fxn=fxn,
        )[:3]
        qg = q.half().reshape(B, H_KV, group, R, D).permute(0, 1, 3, 2, 4).reshape(B * H_KV, R * group, D).contiguous()
        out = Tensor.empty(B, H, R, D, dtype=dtypes.float32, device=q.device)
        fxn = functools.partial(_attend_group, q_first=q_first, rows=rows, kpool=kpool, rdna4=False)
        return Tensor.custom_kernel(out, qg, _bound(cache_kv, q_first, rows), ulist, umask, ulen, fxn=fxn)[0]
    qp = q.half().reshape(B, H_KV, group, R, D).permute(0, 1, 3, 2, 4)
    qp = qp.pad((None, None, None, (0, SLOTS - group), None)).reshape(B * H_KV, R * SLOTS, D).contiguous()
    splits = SPLITS_DECODE if R == 1 else SPLITS_PREFILL if R < LARGE_ROWS else SPLITS_LARGE
    part = Tensor.empty(B * H_KV, splits, R * SLOTS, D, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(B * H_KV, splits, R * SLOTS, 2, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(
        _attend_partial, q_first=q_first, rows=rows, kpool=kpool, topb=topb, rdna4=_is_rdna4(q.device)
    )
    part, stats = Tensor.custom_kernel(part, stats, qp, _bound(cache_kv, q_first, rows), sel, fxn=fxn)[:2]
    out = Tensor.empty(B, H, R, D, dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_attend_combine, q_first=q_first, rows=rows, group=group, h_kv=H_KV, kpool=kpool, topb=topb)
    return Tensor.custom_kernel(out, part, stats, fxn=fxn)[0]


# ---- the model: QSA layers of tinygrad.llm.model.TransformerBlock -------------------------------------------------


def serves(block: Any, x: Tensor, start_pos: tuple[Any, ...]) -> bool:
    """Whether this block's attention runs here: a QSA layer whose indexer selects (the context exceeds what it
    keeps), one position for the batch, on gfx11, with the shapes the kernels are written for."""
    if not _enabled or getattr(block, "indexer", None) is None:
        return False
    cfg, idx = block.config, block.indexer.idx
    if not (idx.qsa and cfg.max_context > idx.top_k and len(start_pos) == 1 and isinstance(x.device, str)):
        return False
    if not amd_custom_kernels_supported(x.device) or _is_rdna4(x.device) or _is_cdna(x.device):
        return False
    group = cfg.n_heads // cfg.n_kv_heads
    return bool(
        cfg.head_dim % (WAVES_N * LANES_PER_WAVE_N) == 0
        and group <= SLOTS
        and cfg.max_context % BLOCK_N == 0
        and idx.head_dim == 4 * WARP_SIZE
        and idx.rope_dim == 2 * WARP_SIZE
        and not idx.rope_interleave
        and BLOCK_N % idx.kpool == 0
        and idx.top_k % idx.kpool == 0
        and not hasattr(block, "attn_sinks")
        and not cfg.sliding_window
    )


def _attention(block: Any, x: Tensor, start_pos: tuple[Any, ...]) -> Tensor:
    """`TransformerBlock._attention` with the indexer's selection and the sparse attention above."""
    cfg, ind = block.config, block.indexer
    idx = ind.idx
    q, k, v = block.attn_q(x), block.attn_k(x), block.attn_v(x)
    if cfg.qk_norm and cfg.qk_norm != cfg.head_dim:
        q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    B, T, _ = x.shape
    gate = None
    if cfg.attn_output_gate:
        qg = q.reshape(B, T, cfg.n_heads, 2, cfg.head_dim)
        q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, cfg.n_heads * cfg.head_dim)
    q = q.reshape(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = k.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
    v = v.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
    if cfg.qk_norm == cfg.head_dim:
        q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    pos = start_pos[0]
    freqs = Tensor.stack(*(block.freqs_cis[p : p + T] for p in start_pos))
    q = llm_model.apply_rope(q[..., : cfg.rope_dim], freqs).cat(q[..., cfg.rope_dim :], dim=-1)
    k = llm_model.apply_rope(k[..., : cfg.rope_dim], freqs).cat(k[..., cfg.rope_dim :], dim=-1)
    assigned_kv = llm_model._update_cache(
        block.cache_kv.transpose(0, 1), Tensor.stack(k, v).transpose(0, 1), start_pos, 3
    ).transpose(0, 1)

    # the indexer: raw keys into its cache, block keys recomputed for the blocks this call fills, queries
    kpool, topb = idx.kpool, idx.top_k // idx.kpool
    kg = llm_model._update_cache(ind.cache_kg, ind.attn_k(x), start_pos, 1)
    R = int(x.max_shape[1])
    weight = ind.k_norm.weight.float()
    blk = update_block_keys(ind.cache_blk, kg, weight, block.freqs_cis, pos, T, R, ind.k_norm.eps, kpool)
    iq = ind.q_norm(ind.attn_q_b(x).reshape(B, T, idx.n_heads, idx.head_dim))
    iq = llm_model.apply_rope(iq.transpose(1, 2), freqs, idx.rope_interleave).transpose(1, 2)
    if isinstance(T, UOp):  # symbolic chunk: pad the rows to the static maximum; garbage rows are sliced off
        iq, q = iq.pad_to((B, R, idx.n_heads, idx.head_dim)), q.pad_to((B, cfg.n_heads, R, cfg.head_dim))
    sel = select(iq.float().contiguous(), blk, pos, T, kpool, topb)
    attn = attend(q, assigned_kv, sel, pos, T, kpool, topb)
    if isinstance(T, UOp):
        attn = attn[:, :, :T]
    attn = attn.transpose(1, 2).reshape(B, T, -1)
    return block.attn_output(attn if gate is None else attn * gate.sigmoid())


def install() -> None:
    """Route QSA layers of tinygrad.llm models here (once per process)."""
    global _enabled
    _enabled = True
    block_cls, indexer_cls = llm_model.TransformerBlock, llm_model.Indexer
    if getattr(block_cls._attention, "_tinyllm_qsa", False):
        return
    attention, indexer_mask, init_state = block_cls._attention, block_cls._indexer_mask, indexer_cls._init_state

    def _attention_hook(self: Any, x: Tensor, start_pos: Any, idx_mask: Tensor | None = None) -> Tensor:
        return _attention(self, x, start_pos) if serves(self, x, start_pos) else attention(self, x, start_pos, idx_mask)

    def _indexer_mask_hook(self: Any, x: Tensor, start_pos: Any, idx_mask: Tensor | None) -> Tensor | None:
        # a served layer selects inside its attention: no mask over the cache is built or passed on
        return idx_mask if serves(self, x, start_pos) else indexer_mask(self, x, start_pos, idx_mask)

    def _init_state_hook(self: Any, x: Tensor) -> None:
        init_state(self, x)
        if self.idx.qsa and not hasattr(self, "cache_blk"):
            blocks = -(-self.max_context // self.idx.kpool)
            device = x.device if isinstance(x.device, str) else None
            self.cache_blk = Tensor.zeros(x.shape[0], self.idx.head_dim, blocks, dtype=dtypes.float32, device=device)
            self.cache_blk = self.cache_blk.contiguous().realize()

    _attention_hook._tinyllm_qsa = True  # type: ignore[attr-defined]
    block_cls._attention = _attention_hook
    block_cls._indexer_mask = _indexer_mask_hook
    indexer_cls._init_state = _init_state_hook
