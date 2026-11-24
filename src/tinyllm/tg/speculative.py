"""Speculative decoding with Qwen3.8's MTP block (plan, phase 3).

One call, a *cycle*, drafts K tokens with the MTP block and verifies them with the main model:

1. The MTP block runs over the K + 1 rows the previous verify produced, pairing the main model's hidden at row j
   with the token the main model chose there. That writes the MTP's KV for the rows that were accepted (the rest
   is overwritten later), and row n of the previous cycle (its last accepted row) yields the first draft.
2. K - 1 more MTP steps, one token each, chain the drafts: each feeds back its own output hidden.
3. The main model runs over [the last accepted token, the K drafts] and samples a token at each of the K + 1 rows
   with the device sampler (`tinyllm.tg.sampler`). The drafts are accepted while they equal the sampled tokens;
   n counts them. The tokens produced are the sampled tokens of rows 0..n: n accepted drafts and one more.

Sampling each row from the main model and stopping at the first disagreement leaves the output distribution
exactly that of plain decoding. More than that: the verify computes, bit for bit, what one-token decoding computes
at each row (below), and each generated position's uniform number depends only on the request's seed and the
position (`tinyllm.engine.sampling.Sampler.device_rows`), so a request produces the same tokens with speculation
on and off, greedy or seeded. `tools/rollback_check.py` checks the bit-for-bit part.

Bit for bit. The projections give the same rows whether a call has one row or several. Everything between them
does not: tinygrad's attention takes a generic float32 path for a few rows where decoding uses the flash decode
kernel with half-precision queries, a norm after a transpose reduces in another order, and the DeltaNet scan
compiled for several tokens fuses multiplies and adds differently. So in a verify (`recording`) the full-attention
and DeltaNet blocks run their projections over all rows and the rest one row at a time with decoding's shapes and
kernels. Without this, greedy outputs differed from plain decoding after about a hundred tokens on near ties.

Rolling back. Attention layers need nothing: a rejected draft's KV entry lies past the accepted position, is
masked out and later overwritten. The recurrent (DeltaNet) layers fold every token into their state, so during
the verify they keep the state after each row in its own buffer (`tinyllm.tg.kernels.deltanet`) and their
convolution window; when fewer than K drafts are accepted, `restore_from` puts back the state after the last
accepted row before the next call.
"""

from __future__ import annotations

import functools
import operator
from typing import Any

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported, flash_attention
from tinygrad.llm.model import GatedDeltaNetBlock, TransformerBlock, apply_rope

from tinyllm.tg.kernels import attention
from tinyllm.tg.kernels.deltanet import gated_delta_step


class SpecDeltaNetBlock(GatedDeltaNetBlock):  # type: ignore[misc]
    """A DeltaNet block that, while `recording` is set, computes each verify row as decoding would and keeps what
    a rollback needs (Qwen3.8's layout only).

    The recording path mirrors `GatedDeltaNetBlock._attention` at the pinned commit for the non-KDA, AMD,
    fixed-length case and must be re-checked on every tinygrad bump; `tools/rollback_check.py` compares it with
    one-token decoding bit for bit.
    """

    recording = False
    snapshots: Any = None  # T - 1 buffers (B, H, V, K) float32: the recurrent state after each row but the last
    conv_window: Any = None  # (B, kernel - 1 + T, channels): the convolution's previous inputs, then the T rows

    def allocate(self, rows: int, device: str, batch: int = 1) -> None:
        """Buffers for verify calls of `rows` tokens."""
        state = (batch, self.num_v_heads, self.head_v_dim, self.head_k_dim)
        self.snapshots = [Tensor.zeros(*state, device=device).contiguous().realize() for _ in range(rows - 1)]
        shape = (batch, self.ssm_conv_kernel - 1 + rows, self.conv_channels)
        self.conv_window = Tensor.zeros(*shape, device=device).contiguous().realize()

    def _attention(self, x: Tensor, start_pos: Any, idx_mask: Tensor | None = None) -> Tensor:
        if not self.recording:
            return super()._attention(x, start_pos, idx_mask)
        B, T, _ = x.shape
        ssm = self.config.ssm
        assert len(start_pos) == 1 and isinstance(T, int) and not ssm.kda, "recording: one fixed-length row"
        pos = start_pos[0]
        assert isinstance(pos, UOp), "recording needs the position as a bound variable"
        initial = Tensor(pos).eq(0)

        x = x.half()
        # the projections give the same rows for one row and for several; everything after them runs one row at
        # a time with decoding's shapes, so it rounds exactly as decoding does (tinyllm.tg.kernels.deltanet)
        qkv, gate_all, beta_all, alpha_all = self.attn_qkv(x), self.attn_gate(x), self.ssm_beta(x), self.ssm_alpha(x)
        a, lb = self.ssm_a.reshape(self.num_v_heads, -1), ssm.gate_lower_bound
        weight = self.ssm_conv1d["weight"].flatten(1)
        width = weight.shape[1]
        state = self.conv_state
        window = initial.where(0, state).cat(qkv.cast(state.dtype), dim=1).contiguous()  # (B, width - 1 + T, C)
        conv_stores = (
            state.uop.store(window[:, T : T + state.shape[1]].uop),
            self.conv_window.uop.store(window.uop),
        )
        recurrent = self.recurrent_state.uop.after(*conv_stores)
        zs = []
        for j in range(T):
            rows = window[:, j : j + width].contiguous()  # the window decoding builds for this row
            conv = functools.reduce(operator.add, (rows[:, i : i + 1] * weight[:, i] for i in range(width))).silu()
            out_gate = gate_all[:, j : j + 1].reshape(B, 1, self.num_v_heads, self.head_v_dim)
            beta = beta_all[:, j : j + 1].sigmoid().reshape(B, 1, self.num_v_heads)
            g = (alpha_all[:, j : j + 1].float() + self.ssm_dt["bias"]).reshape(B, 1, self.num_v_heads, -1)
            log_alpha = g.softplus() * a if lb is None else (g * a.abs()).sigmoid() * lb
            q, k, v = conv.split([self.q_dim, self.q_dim, self.conv_channels - 2 * self.q_dim], dim=-1)
            q, k = (z.reshape(B, 1, self.num_k_heads, self.head_k_dim) for z in (q, k))
            repeat = self.num_v_heads // self.num_k_heads
            q, k = (
                (z.rmsnorm(eps=1e-6 / self.head_k_dim) / self.head_k_dim**0.5).repeat(1, 1, repeat, 1) for z in (q, k)
            )
            v = v.reshape(B, 1, self.num_v_heads, self.head_v_dim)
            q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
            q = q * self.head_k_dim**-0.5
            alpha = log_alpha.transpose(1, 2).exp()
            dst = (self.snapshots[j] if j < T - 1 else self.recurrent_state).uop
            # the position only drives the reset at position 0, which a verify (position >= 1) never takes
            core, recurrent = gated_delta_step(q, k, v, beta, alpha, recurrent, dst, Tensor(pos))
            gate = out_gate.silu() if ssm.silu_gate else out_gate.sigmoid()
            zs.append((self.ssm_norm(core.transpose(1, 2)) * gate).cast(x.dtype).contiguous())
        return self.ssm_out(zs[0].cat(*zs[1:], dim=1).reshape(B, T, -1))

    def restore_from(self, n: int) -> list[Tensor]:
        """Assignments that put back the state after verify row `n` (below rows - 1)."""
        kernel = self.ssm_conv_kernel - 1
        window = self.conv_window[:, n + 1 : n + 1 + kernel]
        return [self.recurrent_state.assign(self.snapshots[n]), self.conv_state.assign(window)]


def recurrent_blocks(model: Any) -> list[SpecDeltaNetBlock]:
    """Turn the model's DeltaNet blocks into `SpecDeltaNetBlock`s (same state and weights) and return them."""
    blocks = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
    for b in blocks:
        b.__class__ = SpecDeltaNetBlock
    return blocks


class SpecAttentionBlock(TransformerBlock):  # type: ignore[misc]
    """A full-attention block that, while `recording` is set, computes each verify row as one-token decoding does:
    the projections over all rows, then per row the norms, RoPE, the key and value store, and attention with the
    flash decode kernel over exactly the keys that row would see there.

    Otherwise tinygrad's tiled attention, which needs multiples of 32 rows, would send a few rows down its generic
    float32 path while decoding uses half-precision queries (about 1e-3 per layer), and the norms and the store
    would round differently for several rows. On AMD one launch of `tinyllm.tg.kernels.attention.flash_decode_rows`
    covers every row, computing each exactly as decoding's kernel would. Called once per row, decoding's kernel
    took 60% of a DFlash2 cycle at 32k positions. Mirrors `TransformerBlock._attention` at the pinned commit
    (re-check on bumps).
    """

    recording = False

    def _attention(self, x: Tensor, start_pos: Any, idx_mask: Tensor | None = None) -> Tensor:
        if not self.recording:
            return super()._attention(x, start_pos, idx_mask)
        cfg = self.config
        assert idx_mask is None and len(start_pos) == 1 and cfg.ssm is not None and not hasattr(self, "attn_sinks")
        B, T, _ = x.shape
        pos = start_pos[0]
        # the projections give the same rows for one row and for several; what follows them (norms after a
        # transpose, RoPE) runs on one row at a time with decoding's shapes, so it rounds as decoding does
        q_all, k_all, v_all = self.attn_q(x), self.attn_k(x), self.attn_v(x)
        qs, ks, vs, gates = [], [], [], []
        for j in range(T):
            q, k, v, gate = self._row(q_all[:, j : j + 1], k_all[:, j : j + 1], v_all[:, j : j + 1], pos + j)
            qs.append(q)
            ks.append(k)
            vs.append(v)
            gates.append(gate)
        # each row's key and value are written as decoding writes them (`_update_cache` for one row): the store
        # kernel takes in the rope, and a store of all rows at once rounds it differently
        cache = self.cache_kv.transpose(0, 1)
        stores = []
        for j in range(T):
            value = Tensor.stack(ks[j], vs[j]).transpose(0, 1)
            target = cache.shrink(tuple((pos + j, pos + j + 1) if i == 3 else None for i in range(cache.ndim)))
            stores.append(target.uop.store(value.cast(cache.dtype).uop))
        kv = Tensor(cache.uop.after(*stores)).transpose(0, 1)
        if amd_custom_kernels_supported(x.device) and attention.supported(qs[0], kv):
            # all rows in one launch, each computed as decoding's kernel computes it (tinyllm.tg.kernels.attention).
            # Each row's query is realized on its own first, as decoding realizes its kernel's input: computed in
            # one kernel for all rows, the norm and RoPE round differently (rows from the third on, at 8 rows).
            hq = [z.half().contiguous() for z in qs]
            q = hq[0].cat(*hq[1:], dim=2)
            attn = attention.flash_decode_rows(q, kv, pos + 1, int(kv.shape[3])).transpose(1, 2).reshape(B, T, -1)
            if gates[0] is not None:
                attn = attn * gates[0].cat(*gates[1:], dim=1).sigmoid()
            return self.attn_output(attn)
        rows = []
        for j in range(T):
            attn = flash_attention(qs[j], kv, pos + (j + 1)).transpose(1, 2).reshape(B, 1, -1)  # decoding's call
            gate = gates[j]
            rows.append(attn if gate is None else attn * gate.sigmoid())
        return self.attn_output(rows[0].cat(*rows[1:], dim=1))

    def _row(self, q: Tensor, k: Tensor, v: Tensor, p: Any) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """`TransformerBlock._attention` from the projections to RoPE for one row at position `p`."""
        cfg, B, T = self.config, q.shape[0], 1
        if cfg.qk_norm and cfg.qk_norm != cfg.head_dim:
            q, k = self.attn_q_norm(q), self.attn_k_norm(k)
        gate = None
        if cfg.attn_output_gate:
            qg = q.reshape(B, T, cfg.n_heads, 2, cfg.head_dim)
            q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, cfg.n_heads * cfg.head_dim)
        q = q.reshape(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = k.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.reshape(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        if cfg.qk_norm == cfg.head_dim:
            q, k = self.attn_q_norm(q), self.attn_k_norm(k)
        freqs = Tensor.stack(self.freqs_cis[p : p + T])
        q = apply_rope(q[..., : cfg.rope_dim], freqs).cat(q[..., cfg.rope_dim :], dim=-1)
        k = apply_rope(k[..., : cfg.rope_dim], freqs).cat(k[..., cfg.rope_dim :], dim=-1)
        return q, k, v, gate


def attention_blocks(model: Any) -> list[SpecAttentionBlock]:
    """Turn the model's full-attention blocks into `SpecAttentionBlock`s and return them."""
    blocks = [b for b in model.blk if type(b) is TransformerBlock]
    for b in blocks:
        b.__class__ = SpecAttentionBlock
    return blocks


class recording:
    """While tracing a verify program, DeltaNet blocks record their snapshots and attention blocks attend row by
    row (`SpecDeltaNetBlock`, `SpecAttentionBlock`)."""

    def __init__(self, blocks: list[Any]) -> None:
        self.blocks = blocks

    def __enter__(self) -> None:
        for b in self.blocks:
            b.recording = True

    def __exit__(self, *exc: object) -> None:
        for b in self.blocks:
            b.recording = False


def leading_matches(sampled: Tensor, drafts: Tensor) -> Tensor:
    """How many leading drafts equal the sampled tokens: `sampled` (K + 1,), `drafts` (K,), both float ids."""
    k = int(drafts.shape[0])
    misses = (sampled[:k] != drafts).cast(dtypes.int32)
    return (misses.cumsum() == 0).cast(dtypes.int32).sum()
