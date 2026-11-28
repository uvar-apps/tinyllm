"""The gated DeltaNet scan with the state read from one buffer and written to another: one token per kernel for a
speculative verify, which keeps the state after each, and a stretch of rows for a prefill call that keeps the
state at a row inside it (`gated_delta_rows`, below).

A speculative verify step runs the main model over the last accepted token and K drafts at once. The recurrent
(DeltaNet) layers fold every token into their state, so when only the first n drafts are accepted the state must
go back to where it was after token n.

Each row runs tinygrad's fused scan (`tinygrad.llm.kernels.amd`, `_gated_delta_prefill_kernel`, re-checked on
every tinygrad bump) for a single token, with one change: the state is read from one buffer and written to
another. Row j reads the state row j - 1 left (the state buffer itself for row 0) and writes snapshot j, its own
buffer; the last row writes the state buffer. With every row reading and writing separate buffers, all rows run
one compiled kernel. A kernel compiled for one token rounds exactly as one-token decoding does,
while the same scan compiled for several tokens differs by a few units in the last place (the compiler fuses
multiplies and adds differently in the token loop). So every snapshot, and the final state, is bit for bit the
state plain decoding would have reached.

Cost: each row reads and writes one state, 3.1 MB per layer on Qwen3.8-27B (48 heads x 128 x 128 floats), so a
verify of 4 rows moves 1.2 GB over its 48 recurrent layers, about 2 ms more than one scan writing snapshots.
"""

from __future__ import annotations

import functools
import itertools

from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.llm.kernels.amd import warp_reduce
from tinygrad.uop.ops import AxisType, KernelInfo, Ops


@functools.cache
def _step_kernel(
    core: UOp,
    q: UOp,
    k: UOp,
    v: UOp,
    beta: UOp,
    alpha: UOp,
    src: UOp,
    kq: UOp,
    dst: UOp,
    start_pos: UOp | None = None,
) -> UOp:
    """Upstream's scan for one token, reading the state from `src` and writing it to `dst` (separate buffers, so
    one compiled kernel serves every row)."""
    batch, heads, tokens, value_dim = core.shape
    row_tile = 4
    key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
    assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim))
    assert key_dim % 32 == 0 and value_dim % row_tile == 0 and tokens == 1
    core, v = (x.reshape(batch * heads, tokens, value_dim) for x in (core, v))
    q, k = (x.reshape(batch * heads, tokens, key_dim) for x in (q, k))
    beta, kq = (x.reshape(batch * heads, tokens) for x in (beta, kq))
    alpha = alpha.reshape(batch * heads, tokens, alpha_dim)
    src, dst = (x.reshape(batch * heads, value_dim, key_dim) for x in (src, dst))

    bh_row = UOp.range(batch * heads * value_dim // row_tile, 0, AxisType.GLOBAL)
    lane = UOp.range(32, 1, axis_type=AxisType.LOCAL)
    bh, row_base = bh_row // (value_dim // row_tile), (bh_row % (value_dim // row_tile)) * row_tile
    rows, cols = tuple(row_base + i for i in range(row_tile)), tuple(lane + i * 32 for i in range(key_dim // 32))
    current = UOp.alloc((row_tile * key_dim // 32,), dtypes.float32, addrspace=AddrSpace.REG)
    initial = None if start_pos is None else start_pos.eq(0)
    current = current.after(
        current.store(
            UOp.stack(
                *(
                    src[bh, row, col].float() if initial is None else initial.where(0, src[bh, row, col].float())
                    for row in rows
                    for col in cols
                )
            )
        )
    )
    token = UOp.range(tokens, 2)
    keys = tuple(k[bh, token, col].load() for col in cols)
    queries = tuple(q[bh, token, col].load() for col in cols)
    updates, stores = [], []
    for row_idx, row in enumerate(rows):
        previous = tuple(current.after(token)[row_idx * key_dim // 32 + i].load() for i in range(key_dim // 32))
        decayed = tuple(
            x * alpha[bh, token, col if alpha_dim > 1 else 0].load() for x, col in zip(previous, cols, strict=True)
        )
        bv = beta[bh, token].load()
        state_k = warp_reduce(
            sum((x * y for x, y in zip(decayed, keys, strict=True)), UOp.const(0, dtypes.float32)), full_wave=True
        )
        state_q = warp_reduce(
            sum((x * y for x, y in zip(decayed, queries, strict=True)), UOp.const(0, dtypes.float32)), full_wave=True
        )
        delta = (v[bh, token, row].load() - state_k) * bv
        updates += [x + delta * y for x, y in zip(decayed, keys, strict=True)]
        stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta * kq[bh, token]))
    step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
    state_stores = (
        dst[bh, row, col].store(current.after(step)[row_idx * key_dim // 32 + i].load().cast(dst.dtype))
        for row_idx, row in enumerate(rows)
        for i, col in enumerate(cols)
    )
    name = "gated_delta_row"
    return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name=name, opts_to_apply=()))


def gated_delta_step(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, alpha: Tensor, src: UOp, dst: UOp, start_pos: Tensor | None
) -> tuple[Tensor, UOp]:
    """One token of `tinygrad.llm.kernels.amd.gated_delta_prefill` (operands shaped as there, T = 1), reading the
    state from the buffer `src` and writing it to the buffer `dst` (UOps of realized buffers). Returns the output
    and `dst` ordered after this kernel, to be the next row's `src`."""
    batch, heads, tokens, key_dim = q.shape
    value_dim = v.shape[-1]
    assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens) and tokens == 1
    assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
    assert key_dim % 32 == 0 and value_dim % 4 == 0
    assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
    if start_pos is not None:
        assert start_pos.uop.is_bound_var
        src = src.after(start_pos.uop)
    dst = dst.after(src)
    core, kq = Tensor.empty_like(v), (q * k).sum(-1).contiguous()
    operands = (x.uop.contiguous() for x in (q, k, v, beta, alpha))
    contig = (core.uop, *operands, src, kq.uop, dst)
    params = tuple(UOp.placeholder_like(x, slot=i) for i, x in enumerate(contig))
    unbound = None if start_pos is None else start_pos.uop.unbound()
    call = _step_kernel(*params, unbound).call(*contig)
    return Tensor(contig[0].after(call)), contig[8].after(call)


@functools.cache
def _rows_kernel(
    core: UOp,
    q: UOp,
    k: UOp,
    v: UOp,
    beta: UOp,
    alpha: UOp,
    src: UOp,
    kq: UOp,
    dst: UOp | None,
    start_pos: UOp | None,
    first: int,
    count: int,
    key_heads: int = 0,
) -> UOp:
    """Upstream's scan over tokens first .. first + count - 1 of the operands, reading the state from `src` and
    writing it to `dst` (None: back to `src`); `core` gets those tokens' outputs and keeps the others. With
    `key_heads`, q, k and kq hold that many heads, which the value heads share in turn (head h reads h % key_heads):
    what upstream repeats into one copy per value head before its scan."""
    batch, heads, tokens, value_dim = core.shape
    row_tile = 4
    key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
    assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim))
    assert key_dim % 32 == 0 and value_dim % row_tile == 0 and first >= 0 and count >= 1 and first + count <= tokens
    assert not key_heads or batch == 1, "shared key heads: one sequence"
    core, v = (x.reshape(batch * heads, tokens, value_dim) for x in (core, v))
    q, k = (x.reshape(key_heads or batch * heads, tokens, key_dim) for x in (q, k))
    beta, kq = beta.reshape(batch * heads, tokens), kq.reshape(key_heads or batch * heads, tokens)
    alpha = alpha.reshape(batch * heads, tokens, alpha_dim)
    src = src.reshape(batch * heads, value_dim, key_dim)
    dst = src if dst is None else dst.reshape(batch * heads, value_dim, key_dim)

    bh_row = UOp.range(batch * heads * value_dim // row_tile, 0, AxisType.GLOBAL)
    lane = UOp.range(32, 1, axis_type=AxisType.LOCAL)
    bh, row_base = bh_row // (value_dim // row_tile), (bh_row % (value_dim // row_tile)) * row_tile
    rows, cols = tuple(row_base + i for i in range(row_tile)), tuple(lane + i * 32 for i in range(key_dim // 32))
    current = UOp.alloc((row_tile * key_dim // 32,), dtypes.float32, addrspace=AddrSpace.REG)
    initial = None if start_pos is None else start_pos.eq(0)
    current = current.after(
        current.store(
            UOp.stack(
                *(
                    src[bh, row, col].float() if initial is None else initial.where(0, src[bh, row, col].float())
                    for row in rows
                    for col in cols
                )
            )
        )
    )
    step_index = UOp.range(count, 2)
    token = step_index + first
    kh = bh % key_heads if key_heads else bh
    keys = tuple(k[kh, token, col].load() for col in cols)
    queries = tuple(q[kh, token, col].load() for col in cols)
    updates, stores = [], []
    for row_idx, row in enumerate(rows):
        previous = tuple(current.after(step_index)[row_idx * key_dim // 32 + i].load() for i in range(key_dim // 32))
        decayed = tuple(
            x * alpha[bh, token, col if alpha_dim > 1 else 0].load() for x, col in zip(previous, cols, strict=True)
        )
        bv = beta[bh, token].load()
        state_k = warp_reduce(
            sum((x * y for x, y in zip(decayed, keys, strict=True)), UOp.const(0, dtypes.float32)), full_wave=True
        )
        state_q = warp_reduce(
            sum((x * y for x, y in zip(decayed, queries, strict=True)), UOp.const(0, dtypes.float32)), full_wave=True
        )
        delta = (v[bh, token, row].load() - state_k) * bv
        updates += [x + delta * y for x, y in zip(decayed, keys, strict=True)]
        stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta * kq[kh, token]))
    step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(step_index)
    state_stores = (
        dst[bh, row, col].store(current.after(step)[row_idx * key_dim // 32 + i].load().cast(dst.dtype))
        for row_idx, row in enumerate(rows)
        for i, col in enumerate(cols)
    )
    name = "gated_delta_shared" if key_heads else "gated_delta_rows"
    return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name=name, opts_to_apply=()))


ROW_LANES = 4  # lanes a state row spans in the prefill scan (`_spans_kernel`); 0: tinygrad's layout, all 32


@functools.cache
def _spans_kernel(
    core: UOp,
    q: UOp,
    k: UOp,
    v: UOp,
    beta: UOp,
    alpha: UOp,
    src: UOp,
    kq: UOp,
    dst: UOp | None,
    start_pos: UOp | None,
    first: int,
    count: int,
    key_heads: int = 0,
    span: int = 4,
) -> UOp:
    """The scan of `_rows_kernel` with each state row on `span` lanes instead of all 32: a wave holds 32 / span
    rows, each lane `key_dim` / span of its row's values.

    tinygrad's layout spreads a row over the whole wave, four values a lane, so a row's dot products with the key
    and the query are each summed across 32 lanes, five swizzles and five adds: 80 of the about 170 vector
    operations per token for a wave of four rows. With a row on 4 lanes a lane sums 32 products itself and a sum
    across the row takes two swizzles: about 21 operations per row and token where the other layout takes 42.

    Measured on Qwen3.8-Flash-Next, ms per 2,048-token call (docs/04-qwen3.8-flash-next.md, section 14): 150 in
    tinygrad's layout, 101 on 4 lanes, 104 on 2, 170 on 8. A row on one lane would need no sum across lanes, but
    its 128 values and the 128 keys do not fit the wave's registers (the compiler spills them: 682).

    The decay is a scalar per head and token (not KDA's per key). The sums run in another order than tinygrad's,
    so the results agree to rounding (`tools/delta_scan_check.py`)."""
    batch, heads, tokens, value_dim = core.shape
    key_dim = q.shape[-1]
    assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim))
    rows_per_wave, per_lane = 32 // span, key_dim // span
    assert span in (2, 4, 8, 16) and key_dim % span == 0 and value_dim % rows_per_wave == 0 and per_lane % 4 == 0
    assert first >= 0 and count >= 1 and first + count <= tokens
    assert not key_heads or batch == 1, "shared key heads: one sequence"
    assert len(alpha.shape) == 3 or alpha.shape[-1] == 1, "one decay per head and token"
    core, v = (x.reshape(batch * heads, tokens, value_dim) for x in (core, v))
    q, k = (x.reshape(key_heads or batch * heads, tokens, key_dim) for x in (q, k))
    beta, kq = beta.reshape(batch * heads, tokens), kq.reshape(key_heads or batch * heads, tokens)
    alpha = alpha.reshape(batch * heads, tokens)
    src = src.reshape(batch * heads, value_dim, key_dim)
    dst = src if dst is None else dst.reshape(batch * heads, value_dim, key_dim)

    tiles = value_dim // rows_per_wave
    bh_tile = UOp.range(batch * heads * tiles, 0, AxisType.GLOBAL)
    # WARP: the wave's own lane, which a sum over a row's lanes needs (lane ^ mask must be the lane the swizzle
    # reads). tinygrad splits a LOCAL range that is used through // span and % span into two hardware dimensions
    # (8 by 4 here), and those sums then mixed four rows; `_rows_kernel` uses its range whole
    lane = UOp.range(32, -1, AxisType.WARP)
    bh, row = bh_tile // tiles, (bh_tile % tiles) * rows_per_wave + lane // span
    kh = bh % key_heads if key_heads else bh
    cols = [(lane % span) * per_lane + c for c in range(per_lane)]
    masks = tuple(m for m in (8, 4, 2, 1) if m < span)  # lane ^ m stays within the row's lanes
    current = UOp.alloc((per_lane,), dtypes.float32, addrspace=AddrSpace.REG)
    initial = None if start_pos is None else start_pos.eq(0)
    loaded = [src[bh, row, col].load().cast(dtypes.float32) for col in cols]
    current = current.after(current.store(UOp.stack(*(x if initial is None else initial.where(0, x) for x in loaded))))
    step_index = UOp.range(count, 2)
    token = step_index + first
    state = [current.after(step_index)[c].load() for c in range(per_lane)]
    keys = [k[kh, token, col].load() for col in cols]
    queries = [q[kh, token, col].load() for col in cols]

    def dot(values: list[UOp]) -> UOp:  # four chains in the lane, then the row's lanes
        chains = [
            sum((state[c] * values[c] for c in range(j + 4, per_lane, 4)), state[j] * values[j]) for j in range(4)
        ]
        return warp_reduce((chains[0] + chains[1]) + (chains[2] + chains[3]), masks=masks)

    decay = alpha[bh, token].load()
    delta = (v[bh, token, row].load() - decay * dot(keys)) * beta[bh, token].load()
    out = core[bh, token, row.valid((lane % span).eq(0))].store(decay * dot(queries) + delta * kq[kh, token].load())
    update = current.store(UOp.stack(*(state[c] * decay + delta * keys[c] for c in range(per_lane))))
    step = UOp.group(out, update).end(step_index)
    stores = [dst[bh, row, col].store(current.after(step)[c].load().cast(dst.dtype)) for c, col in enumerate(cols)]
    name = f"gated_delta_span{span}" + ("_shared" if key_heads else "")
    return UOp.group(*stores).end(lane, bh_tile).sink(arg=KernelInfo(name=name, opts_to_apply=()))


def gated_delta_rows(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha: Tensor,
    state: UOp,
    snapshots: list[tuple[int, UOp]],
    start_pos: Tensor,
    after: tuple[UOp, ...] = (),
    kq: Tensor | None = None,
) -> Tensor:
    """`tinygrad.llm.kernels.amd.gated_delta_prefill` (operands shaped as there) with the recurrent state after
    chosen rows kept: `snapshots` lists (row, buffer) with ascending rows inside the call, and the state after
    row - 1 is left in that buffer. `state` is the state's own buffer (a UOp, like the snapshots'), which the scan
    reads first, after the stores in `after`, and writes last; it is reset at position 0 as upstream's is.

    The scan runs as one kernel per stretch between snapshots, each reading the state from the buffer the
    previous one wrote: a snapshot costs its 3 MB and a kernel launch.

    With `kq` (the per-token dot products of q and k), q and k may hold fewer heads than v: the key heads, which
    upstream repeats for the value heads that share them (`tinyllm.tg.kernels.deltafuse`)."""
    batch, heads, tokens, value_dim = v.shape
    key_heads, key_dim = int(q.shape[1]), int(q.shape[-1])
    shared = key_heads if key_heads != heads else 0
    assert q.shape == k.shape == (batch, key_heads, tokens, key_dim) and beta.shape == (batch, heads, tokens)
    assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
    assert key_dim % 32 == 0 and value_dim % 4 == 0 and (not shared or (kq is not None and heads % key_heads == 0))
    assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
    assert start_pos.uop.is_bound_var
    cuts = [0, *(row for row, _ in snapshots), tokens]
    assert all(a < b for a, b in itertools.pairwise(cuts)), f"snapshot rows {cuts[1:-1]} of {tokens}"
    scalar_decay = len(alpha.shape) == 3 or alpha.shape[-1] == 1
    fits = bool(ROW_LANES) and key_dim % (4 * ROW_LANES) == 0 and value_dim % (32 // ROW_LANES) == 0
    span = ROW_LANES if fits and scalar_decay else 0
    core = Tensor.empty_like(v).uop
    dots = (q * k).sum(-1).contiguous().uop if kq is None else kq.uop
    # a kernel's output is used where it is; anything else is made contiguous
    operands = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in (q, k, v, beta, alpha))
    src = state.after(start_pos.uop, *after)
    for i, first in enumerate(cuts[:-1]):
        last = i == len(snapshots)  # the stretch that writes the state itself: in place when it also read it
        dst = None if last and not snapshots else (state if last else snapshots[i][1]).after(src)
        contig = (core, *operands, src, dots) if dst is None else (core, *operands, src, dots, dst)
        params = tuple(UOp.placeholder_like(x, slot=j) for j, x in enumerate(contig))
        unbound = start_pos.uop.unbound() if i == 0 else None  # a later stretch never starts at position 0
        args = (*params[:8], None if dst is None else params[8], unbound, first, cuts[i + 1] - first, shared)
        kernel = _spans_kernel(*args, span) if span else _rows_kernel(*args)
        call = kernel.call(*contig)
        core, src = contig[0].after(call), contig[6 if dst is None else 8].after(call)
    return Tensor(core)
