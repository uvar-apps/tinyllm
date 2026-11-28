"""Qwen3.8-Flash-Next's gated residual (tinygrad.llm.model.GatedResidual) in prefill: the stream norm and the inject
gates in one kernel.

Each layer keeps four residual streams of 2,560 values per token. Before each sublayer (attention, FFN) tinygrad's
GatedResidual normalises every stream (GroupedRMSNorm), casts the result to fp16 for the down projection, and
projects it to four inject gates on a one-row kernel: separate passes over 10,240 floats a token. Here one
workgroup per token reads the streams once and writes the normalised streams in fp16 (what the down projection
reads), each stream's 1 / rms, and the gates sigmoid(inject(xn) / 4) * 2. The down projection runs on tinyllm's
matrix kernels. The streams are made contiguous once per call and read as views: a contiguous() of a reshape of
them would copy their 84 MB.

The mix. For a Q8_0 or Q6_K up projection of whole 128-row blocks it is formed in the up projection's stores
(`gemm.mix_linear`): the 2,048 x 10,240 fp32 outputs are never written. Calls with a fixed number of rows read the
normalised streams from this kernel's fp16 copy, laid out by tile (`MIX_TILES`): half the bytes of the fp32
streams, in whole tiles, for inputs rounded to fp16 (1,667 -> 1,629 ms a 2,048-token call on code). Other calls
rebuild the fp32 normalised streams from the streams and 1 / rms, as tinygrad computes them.

The residual update. Where one gated residual follows another in a block, the second one's norm kernel forms the
streams itself in those calls (`FUSE_UPDATE`: streams + output * gates, written out and normalised), and the
streams are read once for both (1,640 -> 1,622 ms). The update at a block's end stays tinygrad's: the block returns
the streams. What a pending update is made of is remembered only while the tensor that stands for it is alive
(`_remember`): tinygrad writes out every tensor that is still referenced when it schedules a block, and parts kept
longer cost the pass they were meant to save.

Mirrors `GatedResidual.mix` and `GatedResidual.__call__` at the pinned commit; re-check on every tinygrad bump.
"""

from __future__ import annotations

import functools
import weakref
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.llm import model as llm_model
from tinygrad.llm.kernels.amd import WARP_SIZE, _is_cdna, _is_rdna4, _unbind, amd_custom_kernels_supported, warp_reduce
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from tinyllm.tg.kernels import wmma

THREADS = 256
MIN_TOKENS = 32  # fewer tokens (decoding) keep tinygrad's path
ACTIVE = True  # off: tinygrad's path, for comparison
MIX = True  # the mix in the up projection's stores (`gemm.mix_linear`)
MIX_TILES = True  # that mix reads the norm kernel's fp16 streams by tile, not the fp32 streams again
FUSE_UPDATE = True  # a sublayer's residual update in the next sublayer's norm kernel, where one follows the other
_enabled = False
# What each pending residual update is made of, (streams, output, gates), by the id of the tensor a gated residual
# returned for it. An entry lives as long as that tensor: a tensor alive when tinygrad schedules a block is
# written out there, so parts kept past the tensor would cost the pass over their bytes that fusing saves
_updates: dict[int, tuple[Tensor, Tensor, Tensor]] = {}


@functools.cache
def _norm_inject_kernel(
    xn: UOp,
    gates: UOp,
    invs: UOp,
    *rest: UOp,
    rows: Any,
    count: int,
    eps: float,
    scale: float,
    tile: int = 0,
    update: bool = False,
) -> UOp:
    """rest = (xs, weight, inject). With `tile` (rows a block), xn is laid out by tile as the down projection's
    kernel reads it (`gemm.blocked_shape`): each store, a wave's 32 consecutive values, is one 32-input piece of a
    row.

    With `update`, rest = (new, xs, weight, inject, out, out_gates): the kernel first forms the streams after the
    sublayer before, xs + out * out_gates per stream as `GatedResidual` adds them, writes them to `new`, and
    normalises those. The streams are then read once for the update and the norm."""
    new, (xs, weight, inject), last = (rest[0], rest[1:4], rest[4:]) if update else (None, rest[:3], ())
    rows = _unbind(rows)
    _, width = xs.shape  # (tokens, count * dim) float32
    dim, waves = width // count, THREADS // WARP_SIZE
    per = width // THREADS  # values per thread, `per // count` of them in each stream
    assert dim % THREADS == 0 and per % count == 0
    n_out = inject.shape[0]
    token = UOp.range(rows, 0, AxisType.GLOBAL)
    # the wave and its lane as ranges of their own: tinygrad splits one range of 256 threads that is used through
    # // 32 and % 32 into two hardware dimensions, 8 by 32, and a wave is then 4 lanes of each of 8 "waves"
    wave, lane = UOp.range(waves, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
    tid = wave * WARP_SIZE + lane
    cols = [tid + THREADS * j for j in range(per)]  # coalesced; column c is in stream c // dim
    vals = [xs[token, c].load() for c in cols]
    per_stream = per // count
    first_stores = []
    if (
        new is not None
    ):  # column tid + THREADS * j is dimension tid + THREADS * (j % per_stream) of stream j // per_stream
        out, out_gates = last
        added = [out[token, tid + THREADS * k].load() for k in range(per_stream)]
        factors = [out_gates[token, s].load() for s in range(count)]
        vals = [v + added[j % per_stream] * factors[j // per_stream] for j, v in enumerate(vals)]
        first_stores = [new[token, c].store(v) for c, v in zip(cols, vals, strict=True)]
    lds = UOp.alloc((2, waves, max(count, n_out)), dtypes.float32, addrspace=AddrSpace.LOCAL)

    def block_sums(partials: list[UOp], buf: int) -> list[UOp]:
        reduced = [warp_reduce(p, Ops.ADD, full_wave=True) for p in partials]
        done = lds.after(*(lds[buf, wave, i].store(r) for i, r in enumerate(reduced)))
        return [
            sum((done[buf, w, i].load() for w in range(1, waves)), done[buf, 0, i].load()) for i in range(len(partials))
        ]

    squares = [
        sum(
            (v * v for v in vals[s * per_stream + 1 : (s + 1) * per_stream]),
            vals[s * per_stream] * vals[s * per_stream],
        )
        for s in range(count)
    ]
    inv = [(total / dim + eps).rsqrt() for total in block_sums(squares, 0)]
    normed = [v * inv[j // per_stream] * weight[c].load() for j, (v, c) in enumerate(zip(vals, cols, strict=True))]
    if tile:  # column tid + THREADS * j is element `lane` of input block wave + waves * j
        places = [xn[token // tile, wave + waves * j, token % tile, lane] for j in range(per)]
    else:
        places = [xn[token, c] for c in cols]
    stores = first_stores + [place.store(v.cast(dtypes.half)) for place, v in zip(places, normed, strict=True)]
    dots = [
        sum(
            (v * inject[o, c].load() for v, c in zip(normed[1:], cols[1:], strict=True)),
            normed[0] * inject[o, cols[0]].load(),
        )
        for o in range(n_out)
    ]
    sums = block_sums(dots, 1)
    first = tid.eq(0)
    stores += [
        gates[token, UOp.const(o, dtypes.weakint).valid(first)].store((s * scale).sigmoid() * 2)
        for o, s in enumerate(sums)
    ]
    stores += [invs[token, UOp.const(i, dtypes.weakint).valid(first)].store(v) for i, v in enumerate(inv)]
    name = "hc_update_norm_inject" if update else "hc_norm_inject"
    return UOp.group(*stores).end(wave, lane, token).sink(arg=KernelInfo(name=name, opts_to_apply=()))


def norm_inject(
    block: Any, xs: Tensor, blocked: bool = False, before: tuple[Tensor, Tensor] | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(xn (B, T, count * dim) fp16, gates (B, T, count) and the streams' 1 / rms (B, T, count), float32, and the
    streams) for the streams xs (B, T, count, dim). With `blocked` (one sequence, a fixed number of rows), xn is
    laid out by tile for the down projection (`gemm.blocked_shape`). With `before`, the output (B, T, dim) and the
    gates (B, T, count) of the sublayer before, xs are the streams it started from: the kernel adds its update, and
    the streams returned are the updated ones (a fixed number of rows)."""
    from tinyllm.tg.kernels import gemm

    B, T, count, dim = xs.shape
    R = int(xs.max_shape[1])
    if isinstance(T, UOp):  # a padded copy
        flat = (
            xs.reshape(B, T, count * dim).pad_to((B, R, count * dim)).reshape(B * R, count * dim).float().contiguous()
        )
    else:  # the caller's contiguous streams, viewed: a second contiguous of a reshape would copy them
        flat = xs.float().contiguous().reshape(B * R, count * dim)
    if isinstance(T, UOp):
        flat = Tensor(flat.uop.after(T))  # the kernel's grid follows the bound token count
    tiles = gemm.blocked_shape(B * R, count * dim)
    xn = Tensor.empty(*(tiles if blocked else (B * R, count * dim)), dtype=dtypes.half, device=xs.device)
    gates = Tensor.empty(B * R, count, dtype=dtypes.float32, device=xs.device)
    invs = Tensor.empty(B * R, count, dtype=dtypes.float32, device=xs.device)
    weight = block.norm.weight.float().contiguous()
    inject = block.inject.weight.float().contiguous()
    fxn = functools.partial(
        _norm_inject_kernel,
        rows=B * R if not isinstance(T, UOp) else T,
        count=count,
        eps=block.norm.eps,
        scale=1 / count,
        tile=tiles[2] if blocked else 0,
        update=before is not None,
    )
    streams = xs
    if before is None:
        xn, gates, invs = Tensor.custom_kernel(xn, gates, invs, flat, weight, inject, fxn=fxn)[:3]
    else:
        assert not isinstance(T, UOp), "the update in the norm kernel takes a fixed number of rows"
        out, out_gates = before
        new = Tensor.empty(B * R, count * dim, dtype=dtypes.float32, device=xs.device)
        sources = (out.float().contiguous().reshape(B * R, dim), out_gates.float().contiguous().reshape(B * R, count))
        xn, gates, invs, new = Tensor.custom_kernel(xn, gates, invs, new, flat, weight, inject, *sources, fxn=fxn)[:4]
        streams = new.reshape(B, R, count, dim)
    gates, invs = gates.reshape(B, R, count), invs.reshape(B, R, count)
    if blocked:
        return xn, gates, invs, streams
    xn = xn.reshape(B, R, count * dim)
    if isinstance(T, UOp):
        xn, gates, invs = xn[:, :T], gates[:, :T], invs[:, :T]
    return xn, gates, invs, streams


def mix(block: Any, h: Tensor, xs: Tensor, invs: Tensor, xn: Tensor | None = None) -> Tensor:
    """mean over the streams of sigmoid(up(h)) * xs / rms * weight (B, T, dim) float32, as `GatedResidual.mix`. A Q8_0
    or Q6_K up projection of whole 128-row blocks forms it in the tiled gemm's stores (`gemm.mix_linear`); with
    `xn`, the norm kernel's fp16 streams by tile, from those instead of xs and invs (`MIX_TILES`)."""
    from tinyllm.tg.kernels import gemm, hook

    B, T, count, dim = xs.shape
    weight = block.norm.weight.reshape(count, dim)
    up = block.up
    # the layer's packed weights as the linear hook adopts them on its first call (it replaces `up.weight`)
    kernel = up.__dict__.get("_tinyllm_kernel", hook._UNSET) if MIX else None
    if kernel is hook._UNSET:
        kernel = up._tinyllm_kernel = hook._choose(up)
    R = int(xs.max_shape[1])
    if kernel not in ("q8", "q6k") or (isinstance(T, UOp) and int(B) != 1):  # one sequence: padding comes last
        normed = xs * invs.unsqueeze(-1) * weight
        return (up(h).sigmoid().reshape(xs.shape) * normed).mean(2)
    ggml_type = int(up.ggml_type)
    if isinstance(T, UOp):  # run on the padded maximum, as the linear hook does; the garbage rows are sliced off
        hp, xsp, invp = h.pad_to(h.max_shape), xs.pad_to(xs.max_shape), invs.pad_to(invs.max_shape)
        rows: Any = T
    else:
        hp, xsp, invp, rows = h, xs, invs, None
    hx = hp.reshape(B * R, -1).cast(dtypes.half).contiguous()
    if not gemm.mix_serves(up, hx, ggml_type, count):
        normed = xs * invs.unsqueeze(-1) * weight
        return (up(h).sigmoid().reshape(xs.shape) * normed).mean(2)
    # contiguous first, then the reshape: the other order copies the 84 MB of streams
    flat_xs = xsp.float().contiguous().reshape(B * R, count * dim)
    with wmma.padded_rows(rows):
        out = gemm.mix_linear(
            up,
            up.weight,
            ggml_type,
            hx,
            flat_xs,
            invp.float().contiguous().reshape(B * R, count),
            weight.float().contiguous().reshape(count * dim),
            xn if MIX_TILES and rows is None else None,
        )
    out = out.reshape(B, R, dim)
    return out[:, :T] if isinstance(T, UOp) else out


def _remember(result: Tensor, parts: tuple[Tensor, Tensor, Tensor]) -> None:
    """What the update that `result` stands for is made of, until `result` is gone."""
    key = id(result)
    _updates[key] = parts
    weakref.finalize(result, _updates.pop, key, None)


def serves(block: Any, xs: Tensor) -> bool:
    """A gated residual with an inject projection, one sequence of prefill rows, on gfx11."""
    if not (_enabled and ACTIVE) or not hasattr(block, "inject") or len(xs.shape) != 4 or int(xs.max_shape[0]) != 1:
        return False
    dev = xs.device
    if not isinstance(dev, str) or not amd_custom_kernels_supported(dev) or _is_rdna4(dev) or _is_cdna(dev):
        return False
    count, dim = int(xs.shape[2]), int(xs.shape[3])
    return int(xs.max_shape[1]) >= MIN_TOKENS and dim % THREADS == 0 and (count * dim // THREADS) % count == 0


def install() -> None:
    """Route GatedResidual calls in prefill through `norm_inject` (once per process)."""
    global _enabled
    _enabled = True
    cls = llm_model.GatedResidual
    if getattr(cls.__call__, "_tinyllm_hc", False):
        return
    original = cls.__call__

    def __call__(self: Any, xs: Tensor, sublayer: Any) -> tuple[Tensor, Tensor | None]:
        if not serves(self, xs):
            return original(self, xs, sublayer)  # type: ignore[no-any-return]
        from tinyllm.tg.kernels import gemm

        B, T = xs.shape[:2]
        # the down projection reads its activations by tile when it is a layer of the tiled kernel (gemm.py)
        ggml_type = gemm.takes_blocked(self.down, T, xs.device)
        # those calls (whole blocks of rows) also take the residual update of the sublayer before into the norm
        # kernel; smaller calls compute as they did
        fuse = FUSE_UPDATE and ggml_type is not None
        parts = _updates.pop(id(xs), None) if fuse else None
        before = None
        if parts is not None:  # xs is the sum the gated residual before returned: this norm kernel computes it
            xs, before = parts[0], parts[1:]
        else:
            xs = xs.contiguous()  # the streams once in memory: the norm and the residual update read them
        xn, gates, invs, xs = norm_inject(self, xs, ggml_type is not None, before)
        if ggml_type is None:
            low = self.down(xn)
        else:
            low = gemm.blocked_linear(self.down, ggml_type, xn, int(T)).reshape(B, T, -1)
        mixed = mix(self, (low / self.count).silu(), xs, invs, xn if ggml_type is not None else None)
        out, aux = sublayer(mixed)
        result = xs + out.unsqueeze(2) * gates.unsqueeze(-1)
        if fuse:
            _remember(result, (xs, out, gates))
        return result, aux

    __call__._tinyllm_hc = True  # type: ignore[attr-defined]
    cls.__call__ = __call__
