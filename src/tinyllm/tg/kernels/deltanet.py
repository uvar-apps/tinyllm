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
from tinygrad.uop.ops import AxisType, KernelInfo


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
    dst: UOp,
    start_pos: UOp | None,
    first: int,
    count: int,
) -> UOp:
    """Upstream's scan over tokens first .. first + count - 1 of the operands, reading the state from `src` and
    writing it to `dst`; `core` gets those tokens' outputs and keeps the others."""
    batch, heads, tokens, value_dim = core.shape
    row_tile = 4
    key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
    assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim))
    assert key_dim % 32 == 0 and value_dim % row_tile == 0 and first >= 0 and count >= 1 and first + count <= tokens
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
    step_index = UOp.range(count, 2)
    token = step_index + first
    keys = tuple(k[bh, token, col].load() for col in cols)
    queries = tuple(q[bh, token, col].load() for col in cols)
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
        stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta * kq[bh, token]))
    step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(step_index)
    state_stores = (
        dst[bh, row, col].store(current.after(step)[row_idx * key_dim // 32 + i].load().cast(dst.dtype))
        for row_idx, row in enumerate(rows)
        for i, col in enumerate(cols)
    )
    return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_rows", opts_to_apply=()))


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
) -> Tensor:
    """`tinygrad.llm.kernels.amd.gated_delta_prefill` (operands shaped as there) with the recurrent state after
    chosen rows kept: `snapshots` lists (row, buffer) with ascending rows inside the call, and the state after
    row - 1 is left in that buffer. `state` is the state's own buffer (a UOp, like the snapshots'), which the scan
    reads first, after the stores in `after`, and writes last; it is reset at position 0 as upstream's is.

    The scan runs as one kernel per stretch between snapshots, each reading the state from the buffer the
    previous one wrote: a snapshot costs its 3 MB and a kernel launch."""
    batch, heads, tokens, key_dim = q.shape
    value_dim = v.shape[-1]
    assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens)
    assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
    assert key_dim % 32 == 0 and value_dim % 4 == 0
    assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
    assert start_pos.uop.is_bound_var
    cuts = [0, *(row for row, _ in snapshots), tokens]
    assert all(a < b for a, b in itertools.pairwise(cuts)), f"snapshot rows {cuts[1:-1]} of {tokens}"
    core, kq = Tensor.empty_like(v).uop, (q * k).sum(-1).contiguous().uop
    operands = tuple(x.uop.contiguous() for x in (q, k, v, beta, alpha))
    src = state.after(start_pos.uop, *after)
    for i, first in enumerate(cuts[:-1]):
        dst = (snapshots[i][1] if i < len(snapshots) else state).after(src)
        contig = (core, *operands, src, kq, dst)
        params = tuple(UOp.placeholder_like(x, slot=j) for j, x in enumerate(contig))
        unbound = start_pos.uop.unbound() if i == 0 else None  # a later stretch never starts at position 0
        call = _rows_kernel(*params, unbound, first, cuts[i + 1] - first).call(*contig)
        core, src = contig[0].after(call), contig[8].after(call)
    return Tensor(core)
