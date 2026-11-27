"""Matrix-core (WMMA) linear layer over packed weights, processing several 32-weight groups per loop step.

This is tinygrad's `_quant_linear_wmma` (tinygrad/llm/kernels/amd_gguf.py at the pinned commit) with one change:
its loop over the input dimension handles `groups_per_step` groups of 32 weights per iteration instead of one.
All loads of a step are then issued before its matrix-core work waits on them, so more memory traffic is in
flight at once. With 8 groups per step a step covers exactly one 256-weight super-block, and the position of a
group inside its block becomes a compile-time constant.

Tiling, fragment layout and the final exchange of accumulators are tinygrad's (`_wmma_layout`, `_wmma_words`,
`_wmma_stores`), so any dequantizer written for `_quant_linear_wmma` works here unchanged.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.llm.kernels.amd import WMMA_ARG, _amd_load, _unbind
from tinygrad.llm.kernels.amd_gguf import _amd_byte_perm, _wmma_layout, _wmma_stores, _wmma_words
from tinygrad.uop.ops import AxisType, KernelInfo

Dequant = Callable[[UOp, UOp, int], tuple[UOp, ...] | UOp]
_valid_rows: list[Any] = [None]  # set by the kernel hook around a symbolic chunk's call (`padded_rows`)
SKIP_PADDED_ROWS = True  # False computes the padding rows too (for checks)


@contextlib.contextmanager
def padded_rows(rows: Any) -> Iterator[None]:
    """While a symbolic chunk of `rows` valid rows runs at its padded length: the kernels built meanwhile skip the
    token blocks past `rows` (`valid_rows`)."""
    _valid_rows.append(rows if SKIP_PADDED_ROWS else None)
    try:
        yield
    finally:
        _valid_rows.pop()


def valid_rows() -> Any:
    """The valid rows of the symbolic chunk being built (a bound variable), or None for a whole static chunk."""
    return _valid_rows[-1]


def bind_rows(t: Tensor, rows: Any) -> Tensor:
    """`t` carrying the binding of `rows` into the graph, so the kernel that reads it gets its value."""
    return Tensor(t.uop.after(rows)) if isinstance(rows, UOp) else t


NO_INPUTS = False  # diagnostic: constant activations, no activation loads. Wrong results; shows what loads cost.
SHARED_PAD = 8  # elements of padding per row of a shared activation tile, against local-memory bank conflicts
# experiments: C statements placed at the end of the loop body, for the compiler's scheduling hints, e.g.
# "__builtin_amdgcn_sched_group_barrier(0x20, 16, 0);" (mask, instructions, sync id; they order the code before them)
SCHEDULE: str | None = None

# Packed fp16 conversion of small integer codes, shared by the dequantizers.
# OR-ing an unsigned code u (0..1023) into the bit pattern of 1024.0 gives the fp16 number 1024 + u exactly;
# subtracting a constant then gives the signed weight code exactly, and one multiply applies the scale.
FP16_1024 = 0x6400
PAIR_SELECTORS = (0x0C010C00, 0x0C030C02)  # v_perm_b32: bytes 0,1 (then 2,3) of a word into the two fp16 halves
_CENTER_SCALE = (
    "({{ typedef _Float16 tl_h2 __attribute__((ext_vector_type(2))); "
    "tl_h2 tl_v = __builtin_bit_cast(tl_h2, (unsigned int)({0})) - (tl_h2){{(_Float16)OFFSETf, (_Float16)OFFSETf}}; "
    "__builtin_bit_cast(unsigned int, tl_v * __builtin_bit_cast(tl_h2, (unsigned int)({1}))); }})"
)


_CENTER_AFFINE = (
    "({{ typedef _Float16 tl_h2 __attribute__((ext_vector_type(2))); "
    "tl_h2 tl_v = __builtin_bit_cast(tl_h2, (unsigned int)({0})) - (tl_h2){{(_Float16)OFFSETf, (_Float16)OFFSETf}}; "
    "__builtin_bit_cast(unsigned int, __builtin_elementwise_fma(tl_v, __builtin_bit_cast(tl_h2, (unsigned int)({1})), "
    "__builtin_bit_cast(tl_h2, (unsigned int)({2})))); }})"
)
_SCALE = (
    "({{ typedef _Float16 tl_h2 __attribute__((ext_vector_type(2))); "
    "__builtin_bit_cast(unsigned int, __builtin_bit_cast(tl_h2, (unsigned int)({0})) * "
    "__builtin_bit_cast(tl_h2, (unsigned int)({1}))); }})"
)
# v_perm_b32 with 0x64646464 as the high source: bytes 0,1 (then 2,3) of a word under 0x64, i.e. 1024 + byte
MAGIC_SELECTORS = (0x04010400, 0x04030402)
MAGIC_BYTES = 0x64646464


def byte_pairs(word: UOp) -> tuple[UOp, UOp]:
    """Bytes 0,1 and bytes 2,3 of a word, each pair placed in the low bytes of two 16-bit halves (high bytes zero)."""
    zero = UOp.const(0, dtypes.uint32)
    low, high = (_amd_byte_perm(zero, word, UOp.const(s, dtypes.uint32)) for s in PAIR_SELECTORS)
    return low, high


def magic_pairs(word: UOp) -> tuple[UOp, UOp]:
    """Bytes 0,1 and bytes 2,3 of a word as two packed fp16 numbers 1024 + byte (exact), one v_perm_b32 each."""
    magic = UOp.const(MAGIC_BYTES, dtypes.uint32)
    low, high = (_amd_byte_perm(magic, word, UOp.const(s, dtypes.uint32)) for s in MAGIC_SELECTORS)
    return low, high


def center_scale(pair: UOp, scale2: UOp, offset: int) -> UOp:
    """Two fp16 numbers packed in a word, minus `offset` (exact), times the two fp16 scales in `scale2`."""
    code = _CENTER_SCALE.replace("OFFSET", f"{float(offset)}")
    return UOp.native(code, shape=(), dtype=dtypes.uint32, fmt=True).call(pair, scale2, inline=True)


def center_affine(pair: UOp, scale2: UOp, bias2: UOp, offset: int) -> UOp:
    """Two packed fp16 numbers: (pair - offset) * scale2 + bias2, the subtraction exact, one rounding (fma)."""
    code = _CENTER_AFFINE.replace("OFFSET", f"{float(offset)}")
    return UOp.native(code, shape=(), dtype=dtypes.uint32, fmt=True).call(pair, scale2, bias2, inline=True)


def scale_pair(pair: UOp, scale2: UOp) -> UOp:
    """Two packed fp16 numbers times the two fp16 scales in `scale2` (v_pk_mul_f16)."""
    return UOp.native(_SCALE, shape=(), dtype=dtypes.uint32, fmt=True).call(pair, scale2, inline=True)


def splat_half(value: UOp) -> UOp:
    """An fp16 value (as half, or as its 16 bits) repeated in both halves of a word."""
    bits = (value.bitcast(dtypes.uint16) if value.dtype == dtypes.half else value).cast(dtypes.uint32)
    return bits | (bits << 16)


def wmma_stores(
    out: UOp,
    outputs: tuple,  # type: ignore[type-arg]
    tokens: tuple,  # type: ignore[type-arg]
    accs: tuple,  # type: ignore[type-arg]
    update: UOp,
    lane: UOp,
    wave: UOp,
    output_waves: int,
    kgroups: int,
    lds_bytes: int = 0,
) -> list[UOp]:
    """tinygrad's `_wmma_stores`, with the exchange buffer in local memory (LDS) sized to at least `lds_bytes`.

    The buffer's size limits how many workgroups share a compute unit, and the compiler schedules for that
    occupancy: the fewer waves it can expect, the more registers it lets the loop keep in flight.
    """
    if kgroups > 1 or not lds_bytes:
        return _wmma_stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups)  # type: ignore[no-any-return]
    flat_accs = [acc for output_accs in accs for acc in output_accs]
    per_lane = max(len(flat_accs) * 8, -(-lds_bytes // (4 * output_waves * 32)))
    lds = UOp.alloc((output_waves, 32, per_lane), dtypes.float32, addrspace=AddrSpace.LOCAL)
    stores = [
        lds[wave, lane, a * 8 + i].store(acc.after(update)[i].load())
        for a, acc in enumerate(flat_accs)
        for i in range(8)
    ]
    lds = lds.after(*stores)

    def values(ai: int) -> tuple[UOp, ...]:
        own = tuple(lds[wave, lane, ai * 8 + i].load() for i in range(8))
        peer = tuple(lds[wave, lane ^ 16, ai * 8 + i].load() for i in range(8))
        low = (lane // 16).eq(0)
        return tuple(
            low.where(own[i], peer[i + 4]) if j == 0 else low.where(peer[i], own[i + 4])
            for i in range(4)
            for j in range(2)
        )

    tt = len(tokens)
    return [
        out[token, output].store(value)
        for ot, (output, output_accs) in enumerate(zip(outputs, accs, strict=True))
        for tile, (tile_tokens, _acc) in enumerate(zip(tokens, output_accs, strict=True))
        for token, value in zip(tile_tokens, values(ot * tt + tile), strict=True)
    ]


def wmma_layout(
    out: UOp, out_features: int, token_tile: int, output_tiles: int, lanes: int, waves: int | None = None
) -> tuple:  # type: ignore[type-arg]
    """tinygrad's `_wmma_layout`, or with `waves` waves per workgroup side by side over the outputs.

    tinygrad's layout puts one or two waves in a workgroup. With activations shared through local memory
    (`wmma_linear(share_inputs=True)`) more waves share each copy, so the number is a parameter here.
    """
    if waves is None:
        return _wmma_layout(out, out_features, token_tile, output_tiles, lanes)  # type: ignore[no-any-return]
    if out_features % (16 * output_tiles * waves) or int(out.shape[0]) % token_tile:
        raise ValueError(f"{out_features} outputs, {out.shape[0]} tokens: not whole tiles of {waves} waves")
    token_block = UOp.range(int(out.shape[0]) // token_tile, 0, AxisType.GLOBAL)
    output_block = UOp.range(out_features // (16 * output_tiles * waves), 1, AxisType.GLOBAL)
    lane, wave = UOp.range(lanes, -1, axis_type=AxisType.WARP), UOp.range(waves, 3, axis_type=AxisType.LOCAL)
    col, lane_group, rows = lane % 16, lane // 16, 256 // lanes
    outputs = tuple((output_block * waves + wave) * (16 * output_tiles) + t * 16 + col for t in range(output_tiles))
    inputs = tuple(token_block * token_tile + t * 16 + col for t in range(token_tile // 16))
    tokens = tuple(
        tuple(token_block * token_tile + t * 16 + lane_group * rows + i for i in range(rows))
        for t in range(token_tile // 16)
    )
    return waves, token_block, output_block, lane, wave, lane_group, outputs, inputs, tokens


def wmma_linear(
    out: UOp,
    x: UOp,
    out_features: int,
    in_features: int,
    type_words: int,
    layout: tuple,  # type: ignore[type-arg]
    dequant: Dequant | tuple[Dequant, ...],
    name: str,
    kgroups: int,
    lanes: int,
    groups_per_step: int = 1,
    split_blocks: bool = False,
    input_dtype: DType = dtypes.float16,
    lds_bytes: int = 0,
    share_inputs: bool = False,
    prefetch: tuple[Any, Any] | None = None,
    schedule: str | None = None,
    x_rows: int | None = None,
    steps: UOp | None = None,
    estimates: Any = None,
    rows: Any = None,
    input_row: Callable[[UOp], UOp] | None = None,
    stores: Callable[..., list[UOp]] | None = None,
) -> UOp:
    """out[token, output] = x @ W.T on the matrix cores; `dequant(base, subgroup, half)` gives a lane's weights
    (16 on RDNA3) in `input_dtype` (fp16, or bf16 for the bf16 matrix instruction), as scalars or one vector.

    `type_words` is the number of storage words per 256 weights; `base` addresses the 256-weight block, `subgroup`
    (0..7) its group of 32 and `half` (0, 1) the 16 weights of that group this fragment needs.

    `split_blocks` writes the group index so that tinygrad's range splitting (codegen/simplify.py, a range used
    as `step % c`) turns the loop into an outer loop over 256-weight blocks and an inner loop over the steps of
    a block. Loads that depend only on the block, such as a K-quant's scale words, then move to the outer loop.

    `lds_bytes` reserves local memory per workgroup to lower the occupancy the compiler schedules for (`wmma_stores`).

    `share_inputs` stages each step's activations (token tile x 32 * groups_per_step) in local memory, copied once
    by all threads of the workgroup, and every wave reads its fragments from there; without it each wave loads its
    own copy of every fragment from global memory.

    `prefetch` = (fetch, convert) splits the dequantizer: `fetch(base, subgroup, half)` loads a fragment's raw
    32-bit words and `convert(words, subgroup, half)` turns them into its weights. The words of step s + 1 are
    loaded during step s and held in registers, so their memory latency overlaps a whole step; `dequant` is
    then unused. Experimental, like `share_inputs`: on Q5_K at 128 rows neither beat the plain loop (research
    findings, section 5.5); they are kept for an attempt with the compiler's scheduling hints.

    `schedule` is C placed at the end of the loop body, for the compiler's scheduling hints (`SCHEDULE`, if set,
    replaces it in every kernel).

    `x_rows` is the number of rows of `x` when the layout's rows are not out's (grouped experts gather them).

    `rows` is the number of valid rows of a symbolic chunk computed at its padded length (`valid_rows`): token
    blocks wholly past it run no step (their garbage rows are sliced off).

    `input_row(r)` maps row r (0 .. token tile - 1) of the tile to its row of `x`, for `share_inputs` with gathered
    rows (grouped experts); by default a tile's rows are consecutive rows of `x`.

    `dequant` may be a sequence, one per output tile (two weight matrices for the same outputs). `stores` replaces
    the final stores: stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups) -> stores.

    `steps` replaces the loop's trip count (in_features // 32 // groups_per_step), for a workgroup that has no
    work (0 steps). A trip count read from memory needs `estimates` too: tinygrad's own cost estimate cannot be
    evaluated then, so the kernel carries these and skips tinygrad's optimizer passes (`KernelInfo(optimized=True)`).

    With `share_inputs`, two buffers alternate: while a step computes from one, its
    threads load the next step's activations and write them to the other, so the load latency overlaps the step
    and one barrier per step suffices. RDNA3 fragment layout only (kgroups 1).
    """
    G = groups_per_step
    if (in_features // 32) % G:
        raise ValueError(f"{in_features} inputs are not a whole number of steps of {G} groups")
    if split_blocks and 8 % G:
        raise ValueError(f"split_blocks needs a number of groups per step dividing 8, not {G}")
    x = x.reshape(x_rows or out.shape[0], in_features)
    output_waves, token_block, output_block, lane, wave, lane_group, outputs, input_tokens, tokens = layout
    token_tile, output_tiles = len(tokens) * 16, len(outputs)
    output_words = in_features * type_words // 256
    accs = tuple(
        tuple(UOp.alloc((len(tokens[0]),), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(token_tile // 16))
        for _ in range(output_tiles)
    )
    accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
    if steps is None and rows is not None:
        steps = (token_block * token_tile < _unbind(rows)).where(in_features // 32 // G, 0)
    step = UOp.range(in_features // 32 // G if steps is None else steps, 4)
    wmma_accs = [list(output_accs) for output_accs in accs]
    ks = tuple(word * 4 + b for word in _wmma_words(lane_group, kgroups) for b in range(4))
    a_tile = a_next = None
    a_reads: list[UOp] = []
    if share_inputs and not NO_INPUTS:
        if kgroups != 1:
            raise ValueError("shared activations follow the RDNA3 fragment layout (kgroups 1)")
        width, threads, n_steps = G * 32, output_waves * lanes, in_features // 32 // G
        per_row = width // 8  # 16-byte chunks per row of the tile
        if (token_tile * per_row) % threads:
            raise ValueError(f"a {token_tile} x {width} tile does not split over {threads} threads")
        # Two buffers in local memory and one set of registers per thread form a pipeline: step s computes from
        # buffer s % 2, then writes step s + 1's activations (loaded into registers during step s - 1) to the other
        # buffer and loads step s + 2's into the registers. Every load has a whole step to complete before its use,
        # wherever the compiler places it; one barrier per step (tinygrad adds it) keeps the buffers apart.
        a_lds = UOp.alloc((2, token_tile, width + SHARED_PAD), input_dtype, addrspace=AddrSpace.LOCAL)
        tid, chunks = wave * lanes + lane, token_tile * per_row // threads

        def place(j: int) -> tuple[UOp, UOp]:
            chunk = tid + j * threads
            return chunk // per_row, (chunk % per_row) * 8

        def fetch(j: int, at_step: UOp | int) -> UOp:
            row, column = place(j)
            x_row = token_block * token_tile + row if input_row is None else input_row(row)
            return _amd_load(x[x_row, at_step * width + column], 8)

        def write(target: UOp, buffer: UOp | int, j: int, values: UOp) -> list[UOp]:
            row, column = place(j)
            return [target[buffer, row, column + e].store(values[e].cast(input_dtype)) for e in range(8)]

        a_ready = a_lds.after(*(u for j in range(chunks) for u in write(a_lds, 0, j, fetch(j, 0))))
        held = []
        for j in range(chunks):
            reg = UOp.alloc((8,), input_dtype, addrspace=AddrSpace.REG)
            held.append(reg.after(reg.store(fetch(j, min(1, n_steps - 1)))))
        a_tile = a_ready.after(step)[step % 2]  # the buffers as the previous step left them
        a_next = a_ready, held, n_steps, write, fetch
    col = lane % 16
    held_words: dict[tuple[int, int, int], UOp] = {}
    if prefetch is not None:
        last = in_features // 32 // G - 1

        def pick(fns: Any, output_tile: int) -> Any:  # one function for all output tiles, or one per tile
            return fns[output_tile] if isinstance(fns, (list, tuple)) else fns

        def raw_words(at_step: UOp | int, g: int, half: int, output: UOp, output_tile: int = 0) -> list[UOp]:
            group = at_step * G + g
            fetch_words = pick(prefetch[0], output_tile)
            words: list[UOp] = fetch_words(output * output_words + (group // 8) * type_words, group % 8, half)
            return words

        for g in range(G):
            for half in range(2):
                for output_tile, output in enumerate(outputs):
                    words = raw_words(0, g, half, output, output_tile)
                    reg = UOp.alloc((len(words),), dtypes.uint32, addrspace=AddrSpace.REG)
                    held_words[(g, half, output_tile)] = reg.after(reg.store(UOp.stack(*words)))
    prefetch_stores: list[UOp] = []
    first = True
    steps_per_block = 8 // G if split_blocks else 0
    for g in range(G):
        if steps_per_block == 1:  # a step is a whole block: the group's position in it is a constant
            block, subgroup = step, step.const_like(g)
            group = block * 8 + g
        elif steps_per_block:
            block, subgroup = step // steps_per_block, (step % steps_per_block) * G + g
            group = block * 8 + subgroup
        else:
            group = step * G + g
            block, subgroup = group // 8, group % 8
        for half in range(2):
            if NO_INPUTS:
                afrags = tuple(
                    UOp.stack(*(UOp.const(0.001 * (t + i + 1), input_dtype) for i in range(len(ks))))
                    for t in range(len(input_tokens))
                )
            elif a_tile is not None:
                loads = [
                    [a_tile[t * 16 + col, g * 32 + half * 16 + i].load() for i in ks] for t in range(len(input_tokens))
                ]
                a_reads += [u for frag in loads for u in frag]
                afrags = tuple(UOp.stack(*frag) for frag in loads)
            else:
                afrags = tuple(
                    UOp.stack(*(x[token, group * 32 + half * 16 + i].cast(input_dtype) for i in ks))
                    for token in input_tokens
                )
            for output_tile, output in enumerate(outputs):
                if prefetch is not None:
                    reg = held_words[(g, half, output_tile)]
                    current = reg.after(step)
                    convert = pick(prefetch[1], output_tile)
                    frag = convert([current[i].load() for i in range(int(reg.shape[0]))], subgroup, half)
                    # the last step re-reads step 0
                    upcoming = raw_words((step + 1) % (last + 1), g, half, output, output_tile)
                    read = frag if isinstance(frag, tuple) else (frag,)  # this step's words are used first
                    prefetch_stores.append(reg.after(*read).store(UOp.stack(*upcoming)))
                else:
                    dq = dequant[output_tile] if isinstance(dequant, (list, tuple)) else dequant
                    frag = dq(output * output_words + block * type_words, subgroup, half)
                bfrag = frag if isinstance(frag, UOp) else UOp.stack(*frag)
                for tile, afrag in enumerate(afrags):
                    previous = accs[output_tile][tile].after(step) if first else wmma_accs[output_tile][tile]
                    wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, WMMA_ARG[0], lanes)
            first = False
    next_stores: list[UOp] = []
    if a_next is not None:  # after this step's reads: step + 1 into the other buffer, step + 2 into the registers
        a_ready, held, n_steps, write, fetch = a_next
        current = [reg.after(step) for reg in held]
        target = a_ready.after(*a_reads)
        lds_stores = [u for j, values in enumerate(current) for u in write(target, (step + 1) % 2, j, values)]
        reg_stores = [
            reg.after(*lds_stores).store(fetch(j, (step + 2).minimum(n_steps - 1))) for j, reg in enumerate(held)
        ]
        next_stores = lds_stores + reg_stores
    hints: list[UOp] = []
    schedule = SCHEDULE or schedule
    if schedule:  # a dead store of a value computed after the last matrix instruction carries the statements
        code = "({{ (void)({0}); " + schedule.replace("{", "{{").replace("}", "}}") + " 0; }})"
        anchor = UOp.native(code, shape=(), dtype=dtypes.int32, fmt=True).call(wmma_accs[-1][-1], inline=True)
        hints.append(UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)[0].store(anchor))
    update = UOp.group(
        *(
            acc.store(value)
            for output_accs, values in zip(accs, wmma_accs, strict=True)
            for acc, value in zip(output_accs, values, strict=True)
        ),
        *next_stores,
        *prefetch_stores,
        *hints,
    ).end(step)
    if stores is None:
        stored = wmma_stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups, lds_bytes)
    else:
        stored = stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups)
    info = (
        KernelInfo(name=name, opts_to_apply=())
        if estimates is None
        else KernelInfo(name=name, opts_to_apply=(), estimates=estimates, optimized=True)
    )
    return UOp.group(*stored).end(token_block, output_block, lane, wave).sink(arg=info)
