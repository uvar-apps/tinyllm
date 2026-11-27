"""Qwen3.8-Flash-Next's gated residual (tinygrad.llm.model.GatedResidual) in prefill: the stream norm and the inject
gates in one kernel.

Each layer keeps four residual streams of 2,560 values per token. Before each sublayer (attention, FFN) tinygrad's
GatedResidual normalises every stream (GroupedRMSNorm), casts the result to fp16 for the down projection, and
projects it to four inject gates on a one-row kernel: separate passes over 10,240 floats a token. Here one
workgroup per token reads the streams once and writes the normalised streams in fp16 (what the down projection
reads), each stream's 1 / rms, and the gates sigmoid(inject(xn) / 4) * 2. The down projection and the residual
update stay tinygrad's (and tinyllm's matrix kernels). The mix rebuilds the fp32 normalised streams from the streams
and 1 / rms, as tinygrad's computes them (mixing the fp16 copy instead moved the logits of a 2,600-token prompt by up
to 1.3). For a Q8_0 or Q6_K up projection of whole 128-row blocks it is formed in the up projection's stores
(`gemm.mix_linear`): the 2,048 x 10,240 fp32 outputs are never written, and the logits are bit-identical. The streams
are made contiguous once per call, and the norm and the mix read them as views: a contiguous() of a reshape of them
would copy their 84 MB.

Mirrors `GatedResidual.mix` and `GatedResidual.__call__` at the pinned commit; re-check on every tinygrad bump.
"""

from __future__ import annotations

import functools
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
_enabled = False


@functools.cache
def _norm_inject_kernel(
    xn: UOp, gates: UOp, invs: UOp, xs: UOp, weight: UOp, inject: UOp, rows: Any, count: int, eps: float, scale: float
) -> UOp:
    rows = _unbind(rows)
    _, width = xs.shape  # (tokens, count * dim) float32
    dim, waves = width // count, THREADS // WARP_SIZE
    per = width // THREADS  # values per thread, `per // count` of them in each stream
    assert dim % THREADS == 0 and per % count == 0
    n_out = inject.shape[0]
    token, tid = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(THREADS, 1, AxisType.LOCAL)
    wave = tid // WARP_SIZE
    cols = [tid + THREADS * j for j in range(per)]  # coalesced; column c is in stream c // dim
    vals = [xs[token, c].load() for c in cols]
    per_stream = per // count
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
    stores = [xn[token, c].store(v.cast(dtypes.half)) for v, c in zip(normed, cols, strict=True)]
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
    return UOp.group(*stores).end(tid, token).sink(arg=KernelInfo(name="hc_norm_inject", opts_to_apply=()))


def norm_inject(block: Any, xs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(xn (B, T, count * dim) fp16, gates (B, T, count) and the streams' 1 / rms (B, T, count), float32) for the
    streams xs (B, T, count, dim)."""
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
    xn = Tensor.empty(B * R, count * dim, dtype=dtypes.half, device=xs.device)
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
    )
    xn, gates, invs = Tensor.custom_kernel(xn, gates, invs, flat, weight, inject, fxn=fxn)[:3]
    xn, gates, invs = xn.reshape(B, R, count * dim), gates.reshape(B, R, count), invs.reshape(B, R, count)
    if isinstance(T, UOp):
        xn, gates, invs = xn[:, :T], gates[:, :T], invs[:, :T]
    return xn, gates, invs


def mix(block: Any, h: Tensor, xs: Tensor, invs: Tensor) -> Tensor:
    """mean over the streams of sigmoid(up(h)) * xs / rms * weight (B, T, dim) float32, as `GatedResidual.mix`. A Q8_0
    or Q6_K up projection of whole 128-row blocks forms it in the tiled gemm's stores (`gemm.mix_linear`)."""
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
        )
    out = out.reshape(B, R, dim)
    return out[:, :T] if isinstance(T, UOp) else out


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
        xs = xs.contiguous()  # the streams once in memory: the norm, the mix and the residual update read them
        xn, gates, invs = norm_inject(self, xs)
        mixed = mix(self, (self.down(xn) / self.count).silu(), xs, invs)
        out, aux = sublayer(mixed)
        return xs + out.unsqueeze(2) * gates.unsqueeze(-1), aux

    __call__._tinyllm_hc = True  # type: ignore[attr-defined]
    cls.__call__ = __call__
