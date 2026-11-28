"""Tiled matrix-core GEMM for prefill (gfx11): out = x @ W.T with W dequantized into local memory once per workgroup.

tinygrad's matrix-core loop (`tinyllm.tg.kernels.wmma.wmma_linear`) gives each wave a 64 x 32 tile: every wave
loads its own activation fragments from memory and dequantizes its own weight fragments, 6 fragments for 8 matrix
operations. At 2,048 rows it peaks near 29 TFLOPS with no weights at all and runs Q8_0 at about 20, a third of the
59 the chip's matrix cores reach (tools/prefill_kernel_check.py, docs/04-qwen3.8-flash-next.md section 14).

Here a workgroup of 8 waves computes a BM x BN block (256 x 128, or 256 x 64 for narrow layers; 128 rows when 256
does not divide the call), each wave a 64 x 64 tile held in 16 accumulators. Each step of 32 inputs:
- the threads copy the block's 128 x 32 activations into local memory, and dequantize the block's BN x 32 weights
  into it (one 32-weight block of one output row per thread for Q8_0);
- after a barrier every wave reads 4 activation and 4 weight fragments per 16 inputs from local memory and issues
  16 matrix operations.
Rows of a symbolic chunk past its valid rows skip their steps (`wmma.valid_rows`).

Activations by tile (`blocked_shape`, `blocked_linear`). A step copies 32 inputs of each of the block's 256 rows.
In a (rows, inputs) array those are 256 pieces of 64 bytes, one per row and a row apart: with many inputs and few
outputs (the DeltaNet block's output projection, 6,144 -> 2,560; the gated residual's down projection, 10,240 ->
320) few workgroups share each piece and the kernel waits for memory, 12 to 20 TFLOPS where layers with many
outputs reach 32 (tools/prefill_kernel_check.py, with `DIAG = "hot"` 35). Laid out as (row blocks, input blocks,
256 rows, 32 inputs) a step's activations are 16 contiguous KB and a workgroup's steps follow each other in
memory: the same layers run at 31 to 46. The kernels that produce those activations write them in that layout
(`deltafuse`, `hc`); making it with a copy costs more than it saves.
`mix_linear` is the same loop for Qwen3.8-Flash-Next's gated residual up projection, with stores that form its
mix (`tinyllm.tg.kernels.hc`).

F32 layers (Flash-Next's expert router, 2,560 -> 512) run here too: a thread reads 16 fp32 weights and rounds
them to fp16 as tinygrad's loader does. Unsplit (`UNSPLIT`): 0.36 ms at 2,048 rows against 0.78 on the per-wave
loop (tools/prefill_kernel_check.py --type f32, the conversion of its fp32 input taken off).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.llm.kernels.amd import WMMA_ARG, _amd_load, _is_cdna, _is_rdna4, _unbind
from tinygrad.llm.kernels.amd_gguf import Q6_K, Q8_0, _half, _load_byte, _load_u32
from tinygrad.uop.ops import AxisType, KernelInfo

from tinyllm.tg.kernels.dense import GGML_F32
from tinyllm.tg.kernels.wmma import FP16_1024, bind_rows, byte_pairs, center_scale, magic_pairs, splat_half, valid_rows

BM, BN_WIDE, BN_NARROW = 256, 128, 64  # rows and outputs per workgroup (narrow: layers of 64 k outputs)
BM_SMALL = 128  # rows per workgroup for calls that BM does not divide
WAVE_M, WAVE_N = 64, 64  # the wave tile
PAD = 8  # halves of padding per local-memory row, against bank conflicts
MIN_ROWS = 128  # smaller calls keep the per-wave loop
# Split-K until a call has this many workgroups. Calls of 256-row blocks: 2 per compute unit, which is how many
# run at a time (local memory). On Qwen3.8-Flash-Next at 2,048 rows, 80 against 320: 1,166-1,173 against 1,147-1,149
# tok/s, and the gated residual's down projection takes 0.46 ms in two splits, 0.54 unsplit, 0.58 in eight (the
# sum of the splits included). Calls of 128-row blocks have half the waves per workgroup and keep 320, measured
# on Qwen3.8-27B's 128-row calls.
MIN_WORKGROUPS = 80
MIN_WORKGROUPS_SMALL = 320
# Types that never split. Flash-Next's F32 router (2,560 -> 512: 32 workgroups at 2,048 rows) in the model's
# profile: in four splits the kernel takes 0.34 ms and the sum of the splits 0.20 (16 MB read); unsplit 0.68. By
# the wall clock unsplit wins, 1,721-1,725 against 1,730-1,733 ms a call, and it sums as the per-wave loop does:
# the same logits bit for bit.
UNSPLIT: frozenset[int] = frozenset({GGML_F32})
PIPELINE = False  # double-buffered tiles, the next step fetched into registers: 20 against 25 TFLOPS (Q8_0)
Q8_OFFSET = 1024 + 128
BLOCKED = True  # kernels that can write a layer's activations by tile do (`takes_blocked`)
BLOCKED_K = 0  # experiments: layers with at least this many inputs get their activations copied into tiles first
# diagnostics, wrong results: "inputs" (constant activations, no loads), "hot" (every step reads the call's first
# 32 inputs, so the activations stay cached), "weights" (constant raw words, no weight loads)
DIAG = ""


def _q8_fetch(raw: UOp, row: UOp, block: UOp, half: UOp | int, blocks_per_row: int) -> list[UOp]:
    """A Q8_0 half block's raw words: d, then 16 int8 codes in 4 words."""
    base = (row * blocks_per_row + block) * 17  # halfwords: d, then 32 int8 codes
    # streamed (non-temporal): the weights must not evict the activations from the cache
    codes = [_load_u32(raw, base, 2 + 16 * half + 4 * w, stream=True) for w in range(4)]
    return [raw[base].load().cast(dtypes.uint32), *codes]


def _q8_convert(words: list[UOp]) -> list[UOp]:
    d2 = splat_half(words[0])
    magic = (FP16_1024 | FP16_1024 << 16) | 0x00800080  # the code bytes under 1024's exponent, sign bit flipped
    return [center_scale(pair ^ magic, d2, Q8_OFFSET) for w in words[1:] for pair in byte_pairs(w)]


def _q6k_fetch(raw: UOp, row: UOp, block: UOp, half: UOp | int, blocks_per_row: int) -> list[UOp]:
    """A Q6_K half group: d, the group's int8 scale, and its 16 6-bit codes assembled in 4 words (tinygrad's
    `_quant_word` layout: low 4 bits in ql, high 2 in qh)."""
    sb, sg = block // 8, block % 8
    base = (row * (blocks_per_row // 8) + sb) * 105  # 210-byte super-blocks, in halfwords
    low_shift, high_shift = ((sg % 4 // 2) * 4).cast(dtypes.uint32), ((sg % 4) * 2).cast(dtypes.uint32)
    codes = []
    for w in range(4):
        i = half * 4 + w
        low = _load_u32(raw, base, (sg // 4) * 64 + (sg % 2) * 32 + i * 4, stream=True) >> low_shift
        high = _load_u32(raw, base, 128 + (sg // 4) * 32 + i * 4, stream=True) >> high_shift
        codes.append((low & 0x0F0F0F0F) | ((high & 0x03030303) << 4))
    scale = _load_byte(raw, base, 192 + sg * 2 + half)
    return [raw[base + 104].load().cast(dtypes.uint32), scale, *codes]


def _q6k_convert(words: list[UOp]) -> list[UOp]:
    d, sc = _half(words[0]), words[1].cast(dtypes.uint8).bitcast(dtypes.int8).float()
    s2 = splat_half((d * sc).cast(dtypes.half))
    return [center_scale(pair, s2, 1024 + 32) for w in words[2:] for pair in magic_pairs(w)]


def _f32_fetch(raw: UOp, row: UOp, block: UOp, half: UOp | int, blocks_per_row: int) -> list[UOp]:
    """An F32 half block: 16 weights, one fp32 word each (`dense.find_f32`)."""
    base = (row * blocks_per_row + block) * 32 + half * 16
    loads = [_amd_load(raw[base + 4 * i], 4, stream=True) for i in range(4)]
    return [vec[j] for vec in loads for j in range(4)]


def _f32_convert(words: list[UOp]) -> list[UOp]:
    """Rounded to fp16 as tinygrad's loader rounds an F32 tensor (to nearest), two to a word."""
    bits = [w.bitcast(dtypes.float32).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32) for w in words]
    return [bits[2 * i] | (bits[2 * i + 1] << 16) for i in range(8)]


# a dequantizer: fetch(raw, output row, 32-input block, half) -> raw words, kept in registers a step ahead;
# convert(words) -> the 16 weights of that half block as 8 words of two fp16 each (low half first)
Fetch = Callable[[UOp, UOp, UOp, Any], list[UOp]]
DEQUANT: dict[int, Callable[[int], tuple[Fetch, Callable[[list[UOp]], list[UOp]]]]] = {
    Q8_0: lambda bpr: (lambda raw, row, block, half: _q8_fetch(raw, row, block, half, bpr), _q8_convert),
    Q6_K: lambda bpr: (lambda raw, row, block, half: _q6k_fetch(raw, row, block, half, bpr), _q6k_convert),
    GGML_F32: lambda bpr: (lambda raw, row, block, half: _f32_fetch(raw, row, block, half, bpr), _f32_convert),
}
NAMES = {Q8_0: "q8_0", Q6_K: "q6_k", GGML_F32: "f32"}


@functools.cache
def _gemm_kernel(
    out: UOp,
    raw: UOp,
    x: UOp,
    *extra: UOp,
    out_features: int,
    in_features: int,
    ggml_type: int,
    rows: Any,
    splits: int = 1,
    mix: int = 0,
    blocked: bool = False,
    mix_tiles: bool = False,
) -> UOp:
    """With `blocked`, x is laid out by tile (`blocked_shape`): a step's activations are contiguous in memory.
    With `mix` (the gated residual's stream count, `mix_linear`): extra = (xs, invs, weight), and each wave's
    four 16-output tiles are the same 16 dimensions of the four streams, so the stores can form the mix. With
    `mix_tiles` too, extra = (xn,): the normalised streams in fp16, laid out by tile."""
    M, N, K = int(out.shape[0]) // splits, out_features, in_features
    BM = _bm(M)
    BN = BN_WIDE if N % BN_WIDE == 0 else BN_NARROW
    if mix:
        assert mix == WAVE_N // 16 and BN == BN_WIDE and splits == 1, "a wave's 4 output tiles are the 4 streams"
    WM, WN = BM // WAVE_M, BN // WAVE_N
    TMw, TNw = WAVE_M // 16, WAVE_N // 16
    waves = WM * WN
    threads = waves * 32
    assert M % BM == 0 and N % BN == 0 and K % 32 == 0
    fetch_b, convert_b = DEQUANT[ggml_type](K // 32)
    block_m, block_n = UOp.range(M // BM, 0, AxisType.GLOBAL), UOp.range(N // BN, 1, AxisType.GLOBAL)
    block_k = UOp.range(splits, 2, AxisType.GLOBAL)  # split-K: each split sums its share of the inputs
    wave, lane = UOp.range(waves, 3, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
    tid = wave * 32 + lane
    wm, wn = wave // WN, wave % WN
    col, lane_group = lane % 16, lane // 16
    # two buffers in local memory: step s computes from buffer s % 2 while its threads write step s + 1 (fetched
    # into registers during step s - 1) into the other and fetch step s + 2; one barrier per step
    buffers = 2 if PIPELINE else 1  # local memory per workgroup limits how many share a compute unit
    a_lds = UOp.alloc((buffers, BM, 32 + PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
    b_lds = UOp.alloc((buffers, BN, 16 + PAD // 2), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    accs = [[UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(TNw)] for _ in range(TMw)]
    accs = [[a.after(a.store(a.const_like(0))) for a in row] for row in accs]
    n_steps = K // 32 // splits
    assert not (PIPELINE and splits > 1), "split-K runs the one-buffer loop"
    # a symbolic chunk: blocks of padding rows do nothing
    bound = n_steps if rows is None else (block_m * BM < _unbind(rows)).where(n_steps, 0)
    step = UOp.range(bound, 4)
    a_items, b_items = BM * 4 // threads, BN * 2 // threads  # 8-half chunks of activations, half blocks of weights
    assert BM * 4 % threads == 0 and BN * 2 % threads == 0

    def a_place(r: int) -> tuple[UOp, UOp]:
        chunk = tid + r * threads
        return chunk // 4, (chunk % 4) * 8

    def b_place(r: int) -> tuple[UOp, UOp]:
        item = tid + r * threads
        return item // 2, item % 2

    def fetch_a(r: int, at: UOp | int) -> UOp:
        row, c = a_place(r)
        if DIAG == "inputs":
            return UOp.stack(*(UOp.const(0.5, dtypes.half) for _ in range(8)))
        at = 0 if DIAG == "hot" else at
        if blocked:
            return _amd_load(x[block_m, at, row, c], 8)
        return _amd_load(x[block_m * BM + row, at * 32 + c], 8)

    def weight_row(row: UOp) -> UOp:
        if not mix:
            return block_n * BN + row
        # row = wn * 64 + stream * 16 + c: dimension block_n * 32 + wn * 16 + c of that stream
        dim = N // mix
        return (row // 16 % mix) * dim + block_n * (BN // mix) + row // WAVE_N * 16 + row % 16

    def fetch_w(r: int, at: UOp | int) -> list[UOp]:
        row, h = b_place(r)
        if DIAG == "weights" and ggml_type == GGML_F32:
            return [UOp.const(0x3C800000, dtypes.uint32) for _ in range(16)]
        if DIAG == "weights":
            return [UOp.const(0x3C00 if w == 0 else 0x01020304, dtypes.uint32) for w in range(5)]
        return fetch_b(raw, weight_row(row), at, h)

    def write_a(target: UOp, buf: UOp | int, r: int, vec: Any) -> list[UOp]:
        row, c = a_place(r)
        return [target[buf, row, c + e].store(vec[e]) for e in range(8)]

    def write_b(target: UOp, buf: UOp | int, r: int, words: list[UOp]) -> list[UOp]:
        row, h = b_place(r)
        return [target[buf, row, h * 8 + w].store(v) for w, v in enumerate(convert_b(words))]

    if not PIPELINE:  # one buffer: each step copies its own tiles, then computes (two barriers a step)
        at = block_k * n_steps + step
        stores = [u for r in range(a_items) for u in write_a(a_lds, 0, r, fetch_a(r, at))]
        stores += [u for r in range(b_items) for u in write_b(b_lds, 0, r, fetch_w(r, at))]
        a_tile, b_tile = a_lds.after(*stores)[0], b_lds.after(*stores)[0]
        values = [[a.after(step) for a in row] for row in accs]
        for k16 in range(2):
            afrags = [
                UOp.stack(*(a_tile[wm * WAVE_M + i * 16 + col, k16 * 16 + e].load() for e in range(16)))
                for i in range(TMw)
            ]
            bfrags = [
                UOp.stack(*(b_tile[wn * WAVE_N + j * 16 + col, k16 * 8 + w].load() for w in range(8))).bitcast(
                    dtypes.half
                )
                for j in range(TNw)
            ]
            values = [
                [UOp.wmma(afrags[i], bfrags[j], values[i][j], WMMA_ARG[0], 32) for j in range(TNw)] for i in range(TMw)
            ]
        if mix:
            parts = (block_m, block_n, block_k, wave, lane, wm, wn, col, lane_group)
            return _finish_mix(out, extra, accs, values, step, parts, M, N, mix, ggml_type, mix_tiles)
        return _finish(
            out,
            accs,
            values,
            [],
            step,
            block_m,
            block_n,
            wave,
            lane,
            wm,
            wn,
            col,
            lane_group,
            BN,
            ggml_type,
            block_k,
            M,
        )
    first = [u for r in range(a_items) for u in write_a(a_lds, 0, r, fetch_a(r, 0))]
    first += [u for r in range(b_items) for u in write_b(b_lds, 0, r, fetch_w(r, 0))]
    a_ready, b_ready = a_lds.after(*first), b_lds.after(*first)
    nxt = min(1, n_steps - 1)
    held_a = []
    for r in range(a_items):
        reg = UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG)
        held_a.append(reg.after(reg.store(fetch_a(r, nxt))))
    held_b = []
    for r in range(b_items):
        words = fetch_w(r, nxt)
        reg = UOp.alloc((len(words),), dtypes.uint32, addrspace=AddrSpace.REG)
        held_b.append(reg.after(reg.store(UOp.stack(*words))))
    a_tile, b_tile = a_ready.after(step)[step % 2], b_ready.after(step)[step % 2]
    values = [[a.after(step) for a in row] for row in accs]
    reads: list[UOp] = []
    for k16 in range(2):
        a_loads = [[a_tile[wm * WAVE_M + i * 16 + col, k16 * 16 + e].load() for e in range(16)] for i in range(TMw)]
        b_loads = [[b_tile[wn * WAVE_N + j * 16 + col, k16 * 8 + w].load() for w in range(8)] for j in range(TNw)]
        reads += [u for frag in a_loads + b_loads for u in frag]
        afrags = [UOp.stack(*frag) for frag in a_loads]
        bfrags = [UOp.stack(*frag).bitcast(dtypes.half) for frag in b_loads]
        values = [
            [UOp.wmma(afrags[i], bfrags[j], values[i][j], WMMA_ARG[0], 32) for j in range(TNw)] for i in range(TMw)
        ]
    # after this step's reads: step + 1 into the other buffers, step + 2 into the registers
    a_target, b_target = a_ready.after(*reads), b_ready.after(*reads)
    cur_a, cur_b = [reg.after(step) for reg in held_a], [reg.after(step) for reg in held_b]
    lds_stores = [u for r, reg in enumerate(cur_a) for u in write_a(a_target, (step + 1) % 2, r, reg)]
    lds_stores += [
        u
        for r, reg in enumerate(cur_b)
        for u in write_b(b_target, (step + 1) % 2, r, [reg[i].load() for i in range(int(reg.shape[0]))])
    ]
    upcoming = (step + 2).minimum(n_steps - 1)
    reg_stores = [reg.after(*lds_stores).store(fetch_a(r, upcoming)) for r, reg in enumerate(held_a)]
    reg_stores += [reg.after(*lds_stores).store(UOp.stack(*fetch_w(r, upcoming))) for r, reg in enumerate(held_b)]
    return _finish(
        out,
        accs,
        values,
        [*lds_stores, *reg_stores],
        step,
        block_m,
        block_n,
        wave,
        lane,
        wm,
        wn,
        col,
        lane_group,
        BN,
        ggml_type,
        block_k,
        M,
    )


def _finish_mix(
    out: UOp,
    extra: tuple[UOp, ...],
    accs: list[list[UOp]],
    values: list[list[UOp]],
    step: UOp,
    parts: tuple[UOp, ...],
    M: int,
    N: int,
    mix: int,
    ggml_type: int,
    tiles: bool = False,
) -> UOp:
    """Stores of the gated residual's mix: out[token, d] = mean over the streams s of sigmoid(acc_s) * xs[token, s, d]
    * inv[token, s] * weight[s, d], as `GatedResidual` computes it from the up projection (`hc.py`).

    With `tiles`, the normalised streams come from the norm kernel's fp16 copy, laid out by tile for the down
    projection (`blocked_shape`): a workgroup's 256 rows of 32 dimensions of a stream are one 16 KB tile, where the
    fp32 streams are 256 pieces of 128 bytes, a row (41 KB) apart."""
    block_m, block_n, block_k, wave, lane, wm, wn, col, lane_group = parts
    BM, dim = _bm(M), N // mix
    TMw = len(accs)
    update = UOp.group(*(accs[i][j].store(values[i][j]) for i in range(TMw) for j in range(mix))).end(step)
    stores = []
    d = block_n * (BN_WIDE // mix) + wn * 16 + col
    for i in range(TMw):
        for v in range(8):  # lane l holds rows 2v + l // 16 of column l % 16 of each 16 x 16 tile
            row = wm * WAVE_M + i * 16 + 2 * v + lane_group
            token = block_m * BM + row
            terms = []
            for s in range(mix):
                if tiles:  # (row blocks, 32-input blocks, rows, 32) fp16
                    (xn,) = extra
                    place = xn[block_m, s * (dim // 32) + block_n, row, wn * 16 + col]
                    normed = place.load().cast(dtypes.float32)
                else:  # (M, mix * dim) float32, (M, mix) float32, (mix * dim,) float32
                    xs, invs, weight = extra
                    normed = xs[token, s * dim + d].load() * invs[token, s].load() * weight[s * dim + d].load()
                terms.append(accs[i][s].after(update)[v].load().sigmoid() * normed)
            stores.append(out[token, d].store(sum(terms[1:], terms[0]) / mix))
    sink = UOp.group(*stores).end(wave, lane).end(block_m, block_n, block_k)
    name = f"gemm_{NAMES[ggml_type]}_mix{'_tiles' if tiles else ''}_{BM}x{BN_WIDE}"
    return sink.sink(arg=KernelInfo(name=name, opts_to_apply=()))


def _bm(rows: int) -> int:
    return BM if rows % BM == 0 else BM_SMALL


def _finish(
    out: UOp,
    accs: list[list[UOp]],
    values: list[list[UOp]],
    extra: list[UOp],
    step: UOp,
    block_m: UOp,
    block_n: UOp,
    wave: UOp,
    lane: UOp,
    wm: UOp,
    wn: UOp,
    col: UOp,
    lane_group: UOp,
    BN: int,
    ggml_type: int,
    block_k: UOp,
    M: int,
) -> UOp:
    BM = _bm(M)
    TMw, TNw = len(accs), len(accs[0])
    update = UOp.group(*(accs[i][j].store(values[i][j]) for i in range(TMw) for j in range(TNw)), *extra).end(step)
    # lane l holds rows 2v + l // 16 (v = 0..7) of column l % 16 of each 16 x 16 tile
    stores = []
    for i in range(TMw):
        for j in range(TNw):
            acc = accs[i][j].after(update)
            output = block_n * BN + wn * WAVE_N + j * 16 + col
            for v in range(8):
                token = block_m * BM + wm * WAVE_M + i * 16 + 2 * v + lane_group
                stores.append(out[block_k * M + token, output].store(acc[v].load()))
    name = f"gemm_{NAMES[ggml_type]}_{BM}x{BN}" + (f"_k{block_k.vmax + 1}" if block_k.vmax else "")
    sink = UOp.group(*stores).end(wave, lane).end(block_m, block_n, block_k)
    return sink.sink(arg=KernelInfo(name=name, opts_to_apply=()))


def serves(layer: Any, x: Tensor, ggml_type: int) -> bool:
    """Prefill rows this kernel takes: a type it dequantizes, whole blocks of rows and outputs, on gfx11."""
    if ggml_type not in DEQUANT or _is_rdna4(x.device) or _is_cdna(x.device):
        return False
    tokens, out_features, in_features = int(prod(x.shape[:-1])), int(layer.out_features), int(layer.in_features)
    blocks = 256 if ggml_type == Q6_K else 32  # inputs per weight block
    return tokens >= MIN_ROWS and tokens % BM_SMALL == 0 and out_features % 64 == 0 and in_features % blocks == 0


def mix_serves(layer: Any, x: Tensor, ggml_type: int, count: int) -> bool:
    """Whether `mix_linear` takes this up projection: `serves`, 4 streams, whole 32-dimension blocks per stream."""
    return count == WAVE_N // 16 and serves(layer, x, ggml_type) and int(layer.out_features) % BN_WIDE == 0


def mix_linear(
    layer: Any,
    raw: Tensor,
    ggml_type: int,
    x: Tensor,
    xs: Tensor,
    invs: Tensor,
    weight: Tensor,
    xn: Tensor | None = None,
) -> Tensor:
    """The gated residual's mix for prefill rows (`mix_serves`): mean over the streams of sigmoid(layer(x)) * xs * invs
    * weight, with the up projection's outputs never stored. x (tokens, in) half, xs (tokens, count * dim) and invs
    (tokens, count) float32, weight (count * dim,) float32 -> (tokens, dim) float32. With `xn`, the normalised
    streams in fp16 laid out by tile (`blocked_shape(tokens, count * dim)`), the kernel reads those instead of
    xs, invs and weight: half the bytes, in whole tiles, and the streams' values rounded to fp16."""
    tokens, in_features, out_features = int(x.shape[0]), int(layer.in_features), int(layer.out_features)
    count = int(invs.shape[-1])
    rows = valid_rows()
    tiles = xn is not None
    assert xn is None or (rows is None and tuple(xn.shape) == blocked_shape(tokens, out_features)), "whole tiles"
    fxn = functools.partial(
        _gemm_kernel,
        out_features=out_features,
        in_features=in_features,
        ggml_type=ggml_type,
        rows=rows,
        mix=count,
        mix_tiles=tiles,
    )
    out = Tensor.empty(tokens, out_features // count, dtype=dtypes.float32, device=x.device)
    sources = (xn,) if xn is not None else (xs, invs, weight)
    return Tensor.custom_kernel(out, raw, bind_rows(x, rows), *sources, fxn=fxn)[0]


def blocked_shape(tokens: int, in_features: int) -> tuple[int, int, int, int]:
    """The shape of a call's activations laid out by tile: (row blocks, 32-input blocks, rows of a block, 32).
    Row t, input c of the (tokens, inputs) array is element [t // BM, c // 32, t % BM, c % 32]."""
    bm = _bm(tokens)
    return tokens // bm, in_features // 32, bm, 32


def takes_blocked(layer: Any, tokens: Any, device: Any) -> int | None:
    """The layer's ggml type if a call of `tokens` rows goes to this kernel and it can read the activations by tile
    (`blocked_linear`): a fixed number of rows, a layer the linear hook routes here. None otherwise."""
    return tiled_type(layer, tokens, device) if BLOCKED else None


def tiled_type(layer: Any, tokens: Any, device: Any) -> int | None:
    """The layer's ggml type if the linear hook sends a call of `tokens` rows to this kernel (a fixed number of
    rows of a Q8_0 or Q6_K layer), which then reads the activations in fp16. None otherwise."""
    from tinyllm.tg.kernels import hook, q6k, q8

    if not isinstance(tokens, int) or not isinstance(device, str):
        return None
    kernel = layer.__dict__.get("_tinyllm_kernel", hook._UNSET)
    if kernel is hook._UNSET:  # as the linear hook chooses on a layer's first call (it adopts the packed weights)
        kernel = layer._tinyllm_kernel = hook._choose(layer)
    if not ((kernel == "q8" and q8.GEMM) or (kernel == "q6k" and q6k.GEMM)):
        return None
    ggml_type = int(layer.ggml_type)
    probe = Tensor.empty(tokens, int(layer.in_features), dtype=dtypes.half, device=device)
    return ggml_type if tokens % 16 == 0 and serves(layer, probe, ggml_type) else None


def blocked_linear(layer: Any, ggml_type: int, xb: Tensor, tokens: int) -> Tensor:
    """`layer(x)` for `tokens` rows whose fp16 activations `xb` are laid out by tile (`takes_blocked`,
    `blocked_shape`): (tokens, out) float32."""
    assert xb.dtype == dtypes.half and tuple(xb.shape) == blocked_shape(tokens, int(layer.in_features))
    return _run(layer, layer.weight, ggml_type, xb, tokens, None, True)


def gemm_linear(layer: Any, raw: Tensor, ggml_type: int, x: Tensor) -> Tensor:
    """`layer(x)` for prefill rows (`serves` is true): (..., in) -> (..., out) float32."""
    tokens, in_features = int(prod(x.shape[:-1])), int(layer.in_features)
    rows = valid_rows()
    half = x.cast(dtypes.float16)
    if BLOCKED_K and in_features >= BLOCKED_K and rows is None:  # experiments: a generic copy, slower than it saves
        bm = _bm(tokens)
        xb = half.reshape(tokens // bm, bm, in_features // 32, 32).permute(0, 2, 1, 3).contiguous()
        result = _run(layer, raw, ggml_type, xb, tokens, None, True)
    else:
        xs = bind_rows(half.contiguous().reshape(tokens, in_features), rows)
        result = _run(layer, raw, ggml_type, xs, tokens, rows, False)
    return result.reshape(*x.shape[:-1], int(layer.out_features))


def _run(layer: Any, raw: Tensor, ggml_type: int, xs: Tensor, tokens: int, rows: Any, blocked: bool) -> Tensor:
    """The kernel on `xs`, (tokens, in) fp16 or laid out by tile: (tokens, out) float32, the layer's bias added."""
    in_features, out_features = int(layer.in_features), int(layer.out_features)
    # split-K for layers with few output blocks (the gated residual's down projection, 10,240 -> 320): enough
    # workgroups to fill the machine
    BN = BN_WIDE if out_features % BN_WIDE == 0 else BN_NARROW
    splits, blocks, n_steps = 1, (tokens // _bm(tokens)) * (out_features // BN), in_features // 32
    wanted = MIN_WORKGROUPS if _bm(tokens) == BM else MIN_WORKGROUPS_SMALL
    if ggml_type in UNSPLIT:
        wanted = 1
    while not PIPELINE and blocks * splits < wanted and n_steps % (2 * splits) == 0 and n_steps // splits >= 16:
        splits *= 2
    fxn = functools.partial(
        _gemm_kernel,
        out_features=out_features,
        in_features=in_features,
        ggml_type=ggml_type,
        rows=rows,
        splits=splits,
        blocked=blocked,
    )
    out = Tensor.empty(splits * tokens, out_features, dtype=dtypes.float32, device=xs.device)
    result = Tensor.custom_kernel(out, raw, xs, fxn=fxn)[0]
    if splits > 1:
        result = result.reshape(splits, tokens, out_features).sum(0)
    return result if layer.bias is None else result + layer.bias
