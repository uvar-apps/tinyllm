"""The gated DeltaNet block in prefill: what surrounds its scan, in three kernels.

Around the recurrent scan tinygrad's `GatedDeltaNetBlock._attention` builds the convolution window (the previous
inputs, then the call's), sums the convolution's taps, applies SiLU, splits the result into queries, keys and
values, normalises queries and keys, repeats them for the value heads that share them, converts and transposes
all three, takes the dot product of each query and key, and after the scan normalises the outputs and applies
the output gate. Each of these is a generic kernel that passes over the call's data once: about ten of them, 4 ms a
layer for 2,048 tokens beside the scan's own 4 ms, 147 ms a call on Qwen3.8-Flash-Next
(docs/04-qwen3.8-flash-next.md, section 14).

Here three kernels read the projection's output once and write what the scan reads:

- `delta_qk`, one wave per token and key head: the convolution and SiLU of that head's query and key channels, both
  norms, and their dot product. Queries and keys are written once per key head; the scan reads head h % key_heads
  (`tinyllm.tg.kernels.deltanet`), where tinygrad wrote a copy per value head.
- `delta_v`, one wave per token and value head: the convolution and SiLU of its value channels, written in the
  scan's layout.
- `delta_out`, one wave per token and value head: the RMS norm of the scan's outputs times the output gate.

The convolution's first rows read the previous inputs from the block's state, and nothing builds the window: the
state after the call is the projection's last rows.

`prefill` is the block's `_attention` for a call of a fixed number of tokens, with the two things a padded call
needs (`tinyllm.tg.speculative.padded`): rows past `real` leave the recurrent state alone, and the state after row
`mid` - 1 is kept. Mirrors the pinned commit for the non-KDA layout; re-check on every tinygrad bump.
`tools/deltanet_check.py` compares it with tinygrad's own path on a model's blocks.
"""

from __future__ import annotations

import functools
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm import model as llm_model
from tinygrad.llm.kernels.amd import WARP_SIZE, amd_custom_kernels_supported, warp_reduce
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from tinyllm.tg.kernels import gemm
from tinyllm.tg.kernels.deltanet import gated_delta_rows

MIN_TOKENS = 32  # fewer rows (decoding, a speculative verify) keep their own paths
ACTIVE = True  # off: tinygrad's kernels around the scan, for comparison
_enabled = False


def serves(block: Any, x: Tensor) -> bool:
    """Whether `prefill` takes this call: a fixed number of rows of one sequence, the non-KDA layout, head sizes
    that fill whole waves."""
    ssm = block.config.ssm
    B, T, _ = x.shape
    return bool(
        ACTIVE
        and _enabled
        and isinstance(T, int)
        and T >= MIN_TOKENS
        and B == 1
        and ssm is not None
        and not ssm.kda
        and block.head_k_dim % WARP_SIZE == 0
        and block.head_v_dim % WARP_SIZE == 0
        and isinstance(x.device, str)
        and amd_custom_kernels_supported(x.device)
    )


def _conv(qkv: UOp, state: UOp, weight: UOp, initial: UOp, token: UOp, channel: UOp, hist: int) -> UOp:
    """SiLU of the convolution at one token and channel: tap i reads the input `hist` - i rows back, from the call's
    rows or, before its first row, from the previous inputs in `state` (zero at position 0)."""
    total = None
    for i in range(hist + 1):
        row = token + (i - hist)
        inside = row >= 0
        here = qkv[inside.where(row, 0), channel].load()
        before = initial.where(0, state[inside.where(0, row + hist), channel].load())
        term = inside.where(here, before) * weight[channel, i].load().cast(dtypes.float32)
        total = term if total is None else total + term
    assert total is not None
    return total.silu()


@functools.cache
def _qk_kernel(
    qn: UOp, kn: UOp, kq: UOp, qkv: UOp, state: UOp, weight: UOp, start: UOp, heads: int, dim: int, hist: int
) -> UOp:
    """Queries and keys of every token and key head: convolution, SiLU, the l2 norm tinygrad writes as an RMS norm
    over the head divided by sqrt(dim), the queries' further 1 / sqrt(dim), and the dot product of the two."""
    tokens = int(qkv.shape[0])
    qn, kn, kq = qn.reshape(heads, tokens, dim), kn.reshape(heads, tokens, dim), kq.reshape(heads, tokens)
    token, head = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(heads, 1, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, 2, AxisType.LOCAL)
    initial = start.eq(0)
    cols = [lane + i * WARP_SIZE for i in range(dim // WARP_SIZE)]
    qs = [_conv(qkv, state, weight, initial, token, head * dim + c, hist) for c in cols]
    ks = [_conv(qkv, state, weight, initial, token, (heads + head) * dim + c, hist) for c in cols]

    def scale(values: list[UOp], extra: float) -> list[UOp]:
        squares = warp_reduce(sum((v * v for v in values[1:]), values[0] * values[0]), Ops.ADD, full_wave=True)
        inv = (squares / dim + 1e-6 / dim).rsqrt()
        return [v * inv / dim**0.5 * extra if extra != 1.0 else v * inv / dim**0.5 for v in values]

    qs, ks = scale(qs, dim**-0.5), scale(ks, 1.0)
    dot = warp_reduce(sum((a * b for a, b in zip(qs[1:], ks[1:], strict=True)), qs[0] * ks[0]), Ops.ADD, full_wave=True)
    stores = [qn[head, token, c].store(v) for v, c in zip(qs, cols, strict=True)]
    stores += [kn[head, token, c].store(v) for v, c in zip(ks, cols, strict=True)]
    stores.append(kq[head, token.valid(lane.eq(0))].store(dot))
    return UOp.group(*stores).end(lane, head, token).sink(arg=KernelInfo(name="delta_qk", opts_to_apply=()))


@functools.cache
def _v_kernel(v: UOp, qkv: UOp, state: UOp, weight: UOp, start: UOp, first: int, hist: int) -> UOp:
    """Values of every token and value head, in the scan's layout: convolution and SiLU of the channels from
    `first` on."""
    heads, tokens, dim = (int(n) for n in v.shape[-3:])
    v = v.reshape(heads, tokens, dim)
    token, head = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(heads, 1, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, 2, AxisType.LOCAL)
    initial = start.eq(0)
    stores = []
    for i in range(dim // WARP_SIZE):
        row = lane + i * WARP_SIZE
        stores.append(
            v[head, token, row].store(_conv(qkv, state, weight, initial, token, first + head * dim + row, hist))
        )
    return UOp.group(*stores).end(lane, head, token).sink(arg=KernelInfo(name="delta_v", opts_to_apply=()))


@functools.cache
def _out_kernel(z: UOp, core: UOp, gate: UOp, weight: UOp, eps: float, silu: bool, tile: int = 0) -> UOp:
    """The block's output before its projection: the scan's outputs under an RMS norm over each head, times the
    norm's weight and the activated output gate, in fp16. With `tile` (rows a block), z is laid out by tile as the
    output projection's kernel reads it (`gemm.blocked_shape`): each store is one 32-input piece of a row."""
    heads, tokens, dim = (int(n) for n in core.shape[-3:])
    core = core.reshape(heads, tokens, dim)
    token, head = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(heads, 1, AxisType.GLOBAL)
    lane = UOp.range(WARP_SIZE, 2, AxisType.LOCAL)
    rows = [lane + i * WARP_SIZE for i in range(dim // WARP_SIZE)]
    values = [core[head, token, r].load() for r in rows]
    squares = warp_reduce(sum((v * v for v in values[1:]), values[0] * values[0]), Ops.ADD, full_wave=True)
    inv = (squares / dim + eps).rsqrt()
    stores = []
    for i, (value, r) in enumerate(zip(values, rows, strict=True)):
        g = gate[token, head * dim + r].load()
        act = g.silu() if silu else g.sigmoid()
        out = value * inv * weight[r].load().cast(dtypes.float32) * act
        place = (
            z[token // tile, head * (dim // WARP_SIZE) + i, token % tile, lane] if tile else z[token, head * dim + r]
        )
        stores.append(place.store(out.cast(dtypes.half)))
    return UOp.group(*stores).end(lane, head, token).sink(arg=KernelInfo(name="delta_out", opts_to_apply=()))


def prefill(block: Any, x: Tensor, start_pos: Any, real: Any = None, mid: int = 0) -> Tensor:
    """`GatedDeltaNetBlock._attention` for a call `serves` accepts. `real` (a bound variable): the rows before it
    are the prompt's and the others padding, exact no-ops for the recurrent state, as tinygrad makes the padding
    of a symbolic chunk (beta 0, decay 1); the convolution state is the one after row `real` - 1. `mid`: the
    recurrent and convolution state after row `mid` - 1 is kept in `block.mid_state` and `block.mid_conv`
    (0: not). Both rows must be at least the convolution's reach into the call."""
    B, T, _ = x.shape
    ssm = block.config.ssm
    assert len(start_pos) == 1 and isinstance(T, int) and B == 1 and not ssm.kda
    pos = start_pos[0]
    assert isinstance(pos, UOp), "the position must be a bound variable"
    heads, v_dim, k_heads, k_dim = block.num_v_heads, block.head_v_dim, block.num_k_heads, block.head_k_dim
    hist, channels = block.ssm_conv_kernel - 1, block.conv_channels
    assert hist <= T and (not mid or mid >= hist)

    x = x.half()
    gate = block.attn_gate(x).float().contiguous().reshape(T, heads * v_dim)
    beta = block.ssm_beta(x).sigmoid().reshape(B, T, heads)
    g = (block.ssm_alpha(x).float() + block.ssm_dt["bias"]).reshape(B, T, heads, -1)
    a, lb = block.ssm_a.reshape(heads, -1), ssm.gate_lower_bound
    log_alpha = g.softplus() * a if lb is None else (g * a.abs()).sigmoid() * lb
    if real is not None:  # padding rows: beta 0 and decay 1
        keep = Tensor.arange(T).reshape(1, T, 1) < Tensor(real)
        beta, log_alpha = keep.where(beta, 0), keep.unsqueeze(-1).where(log_alpha, 0)
    beta, alpha = beta.transpose(1, 2).float(), log_alpha.transpose(1, 2).exp()

    state = block.conv_state
    qkv = block.attn_qkv(x).float().contiguous().reshape(T, channels)
    weight = block.ssm_conv1d["weight"].flatten(1).contiguous()
    position, old = Tensor(pos), state.reshape(hist, channels)
    qn = Tensor.empty(B, k_heads, T, k_dim, dtype=dtypes.float32, device=x.device)
    kn, kq = Tensor.empty_like(qn), Tensor.empty(B, k_heads, T, dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(_qk_kernel, heads=k_heads, dim=k_dim, hist=hist)
    qn, kn, kq = Tensor.custom_kernel(qn, kn, kq, qkv, old, weight, position, fxn=fxn)[:3]
    v = Tensor.empty(B, heads, T, v_dim, dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(_v_kernel, first=2 * k_heads * k_dim, hist=hist)
    v = Tensor.custom_kernel(v, qkv, old, weight, position, fxn=fxn)[0]

    # the previous inputs after the call are its last rows (after the last real row), whatever they were before.
    # Nothing orders a write of the state after the kernels above, which read it as it was (tinygrad's own store
    # reads the window it built from the state). So the stores read the projection as a later state of its buffer,
    # one that follows the first kernel; the scheduler then runs every reader of the earlier state, the second
    # kernel too, before it. Naming both kernels there would order each before the other
    after = Tensor(qkv.uop.after(kq.uop))
    end = T if real is None else real
    stores = [state.uop.store(after[end - hist : end].reshape(B, hist, channels).cast(state.dtype).uop)]
    if mid:
        rows = after[mid - hist : mid].reshape(B, hist, channels).cast(state.dtype)
        stores.append(block.mid_conv.uop.store(rows.uop))

    snapshots = [(mid, block.mid_state.uop)] if mid else []
    stores_ = tuple(stores)
    core = gated_delta_rows(qn, kn, v, beta, alpha, block.recurrent_state.uop, snapshots, position, stores_, kq)
    norm = block.ssm_norm
    # the output projection reads its activations by tile when it is a layer of the tiled kernel (gemm.py)
    ggml_type = gemm.takes_blocked(block.ssm_out, T, x.device)
    tiles = gemm.blocked_shape(T, heads * v_dim)
    z = Tensor.empty(*((T, heads * v_dim) if ggml_type is None else tiles), dtype=dtypes.half, device=x.device)
    tile = 0 if ggml_type is None else tiles[2]
    fxn = functools.partial(_out_kernel, eps=norm.eps, silu=bool(ssm.silu_gate), tile=tile)
    z = Tensor.custom_kernel(z, core, gate, norm.weight.contiguous(), fxn=fxn)[0]
    if ggml_type is None:
        return block.ssm_out(z.cast(x.dtype).reshape(B, T, -1))
    return gemm.blocked_linear(block.ssm_out, ggml_type, z, T).reshape(B, T, -1)


def install() -> None:
    """Route the prefill calls of tinygrad.llm's DeltaNet blocks here (once per process)."""
    global _enabled
    _enabled = True
    block_cls = llm_model.GatedDeltaNetBlock
    original = block_cls._attention
    if getattr(original, "_tinyllm_delta", False):
        return

    def _attention(self: Any, x: Tensor, start_pos: Any, idx_mask: Tensor | None = None) -> Tensor:
        if len(start_pos) == 1 and isinstance(start_pos[0], UOp) and serves(self, x):
            return prefill(self, x, start_pos)
        return original(self, x, start_pos, idx_mask)

    _attention._tinyllm_delta = True  # type: ignore[attr-defined]
    block_cls._attention = _attention
