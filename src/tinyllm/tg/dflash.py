"""DFlash2, the block-diffusion drafter for Qwen3.8-27B (plan, phase 3d).

A separate GGUF (`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, architecture `dflash`, 1.1 GB) that proposes a whole block of
drafts in one pass. It reads the target's residual stream at the inputs of five target layers (`target_layers`)
and borrows the target's token embedding and output head. Structure, from llama.cpp (`src/models/dflash.cpp`,
`dflash2.cpp`, and the driver in `common/speculative.cpp`):

* Injection. For every token the target has processed, the five layer inputs are concatenated, fused by `fc` and
  `enc.output_norm`, and each draft layer writes keys (normed, roped) and values of that row into its own cache at
  the token's position. No queries.
* Drafting. A block [last sampled token, mask x (block_size - 1)] at the next positions runs through the layers
  with non-causal attention over the injected context and the block itself, in a sliding window. Each layer wraps
  attention and the FFN in a short depthwise convolution over the block, with per-token weights from
  `*_conv_proj` added to a static `*_conv_base`. The final norm feeds the target's head.
* Selection (DFlash2). Per block position the head's 16 best tokens are scored pairwise against the previous
  position's: unary logit + successor[token] . (predecessor[previous token] * gate(hidden)), with a rank-256
  gate; a greedy walk from the anchor picks one candidate per position. Those are the drafts.
"""

from __future__ import annotations

import functools
import math
import operator
from pathlib import Path
from typing import Any

from tinygrad import Device, Tensor, UOp, dtypes, nn
from tinygrad.llm.gguf import gguf_parse, gguf_shard
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported
from tinygrad.llm.model import apply_rope, precompute_freqs_cis

from tinyllm.tg.kernels.topk import topk


def _top(row: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """The k largest values of a 1-D row and their ids (int32): tinyllm's kernel on AMD, tinygrad's sort elsewhere."""
    if amd_custom_kernels_supported(row.device):
        return topk(row, k)
    values, ids = row.topk(k)
    return values, ids.cast(dtypes.int32)


class _Layer:
    def __init__(
        self, dim: int, heads: int, kv_heads: int, head_dim: int, hidden: int, conv: int, group: int, eps: float
    ) -> None:
        self.attn_norm, self.ffn_norm = nn.RMSNorm(dim, eps), nn.RMSNorm(dim, eps)
        self.attn_q = Linear(dim, heads * head_dim, bias=False)
        self.attn_k = Linear(dim, kv_heads * head_dim, bias=False)
        self.attn_v = Linear(dim, kv_heads * head_dim, bias=False)
        self.attn_output = Linear(heads * head_dim, dim, bias=False)
        self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(head_dim, eps), nn.RMSNorm(head_dim, eps)
        self.ffn_gate, self.ffn_up = Linear(dim, hidden, bias=False), Linear(dim, hidden, bias=False)
        self.ffn_down = Linear(hidden, dim, bias=False)
        self.attn_conv_proj, self.ffn_conv_proj = Linear(dim, conv, bias=False), Linear(dim, conv, bias=False)
        kernel = conv // (2 * (dim // group))
        self.attn_conv_base = Tensor.zeros(2, kernel, dim)  # (side: before, after) x tap x channel
        self.ffn_conv_base = Tensor.zeros(2, kernel, dim)


class _Encoder:
    def __init__(self, dim: int, eps: float) -> None:
        self.output_norm = nn.RMSNorm(dim, eps)


class DFlash:
    """The drafter's weights and its own KV cache; `inject` and `draft` build graphs for the caller to run."""

    def __init__(self, kv: dict[str, Any], max_context: int) -> None:
        a = "dflash"
        self.dim, self.layers = int(kv[f"{a}.embedding_length"]), int(kv[f"{a}.block_count"])
        self.heads, self.kv_heads = int(kv[f"{a}.attention.head_count"]), int(kv[f"{a}.attention.head_count_kv"])
        self.head_dim = int(kv[f"{a}.attention.key_length"])
        self.eps = float(kv[f"{a}.attention.layer_norm_rms_epsilon"])
        self.block_size = int(kv[f"{a}.block_size"])
        self.kernel, self.group = int(kv[f"{a}.conv_kernel_size"]), int(kv[f"{a}.conv_group_size"])
        self.rank, self.top_k = int(kv[f"{a}.selector_rank"]), int(kv[f"{a}.selector_top_k"])
        self.target_layers = tuple(int(i) for i in kv[f"{a}.target_layers"])
        self.window = int(kv.get(f"{a}.attention.sliding_window", 0) or max_context)
        self.mask_token = int(kv["tokenizer.ggml.mask_token_id"])
        self.rope_theta = float(kv[f"{a}.rope.freq_base"])
        assert not kv.get(f"{a}.attention.causal", False), "DFlash2 attends without a causal mask"
        conv = 2 * self.kernel * (self.dim // self.group)
        hidden = int(kv[f"{a}.feed_forward_length"])
        self.blk = [
            _Layer(self.dim, self.heads, self.kv_heads, self.head_dim, hidden, conv, self.group, self.eps)
            for _ in range(self.layers)
        ]
        self.fc = Linear(len(self.target_layers) * self.dim, self.dim, bias=False)
        self.enc, self.output_norm = _Encoder(self.dim, self.eps), nn.RMSNorm(self.dim, self.eps)
        self.selector_hidden = Linear(self.dim, self.rank, bias=False)
        self.selector_predecessor = nn.Embedding(1, self.rank)
        self.selector_successor = nn.Embedding(1, self.rank)
        # the cache holds position p at index p + pad, so a fixed-size window ending at any position is one slice
        self.span = self.window + self.block_size  # keys a block attends to: the window before its first row, and it
        self.pad = self.span
        self.max_context = max_context
        self.cache: list[Tensor] = []
        self.freqs_cis: Tensor | None = None

    # ---- loading ----

    @staticmethod
    def load(path: str | Path, max_context: int) -> DFlash:
        kv, entries = gguf_parse(str(path))
        if kv.get("general.architecture") != "dflash" or "selector_hidden.weight" not in entries:
            raise ValueError(f"{path} is not a DFlash2 drafter")
        state = gguf_shard(entries, (Device.DEFAULT,))
        state = {k: v.cast(dtypes.float16) if dtypes.is_float(v.dtype) else v for k, v in state.items()}
        model = DFlash(kv, max_context)
        for name in ("selector_predecessor", "selector_successor"):  # vocabulary-sized: shaped from the file
            getattr(model, name).weight = Tensor.zeros(*state[f"{name}.weight"].shape)
        nn.state.load_state_dict(model, state, verbose=False, consume=True, realize=False)
        # gathered by token id every draft: decode the codebooks once
        for name in ("selector_predecessor", "selector_successor"):
            table = getattr(model, name)
            table.weight = table.weight.contiguous().realize()
        for layer in model.blk:
            layer.attn_conv_base = layer.attn_conv_base.float().contiguous().realize()
            layer.ffn_conv_base = layer.ffn_conv_base.float().contiguous().realize()
        return model

    def _init_state(self, device: Any) -> None:
        if not self.cache:
            length = self.max_context + self.pad
            shape = (2, 1, self.kv_heads, length, self.head_dim)
            self.cache = [
                Tensor.zeros(*shape, dtype=dtypes.half, device=device).contiguous().realize() for _ in self.blk
            ]
            end = self.max_context + self.block_size
            self.freqs_cis = precompute_freqs_cis(self.head_dim, end, self.rope_theta, device=device).realize()

    # ---- pieces ----

    def _rope(self, x: Tensor, start: Any, rows: int) -> Tensor:
        """x (B, heads, rows, head_dim) at positions start..start + rows - 1."""
        assert self.freqs_cis is not None
        return apply_rope(x, self.freqs_cis[start : start + rows].unsqueeze(0))

    def _store(self, i: int, k: Tensor, v: Tensor, start: Any) -> UOp:
        """Write keys and values (B, kv_heads, rows, head_dim) of positions start.. into layer i's cache (rows may
        be symbolic: a prefill chunk of variable length)."""
        cache, rows = self.cache[i], k.shape[2]
        index = start + self.pad
        target = cache.shrink((None, None, None, (index, index + rows), None))
        return target.uop.store(Tensor.stack(k, v).cast(cache.dtype).uop)

    def _conv(self, hidden: Tensor, dynamic: Tensor, base: Tensor, side: int) -> Tensor:
        """The block convolution: hidden (B, T, D) mixed with its previous rows, weights per token and channel."""
        B, T, D = hidden.shape
        groups = D // self.group
        coeff = dynamic.reshape(B, T, 2, self.kernel, groups)[:, :, side]  # (B, T, kernel, groups)
        coeff = coeff.unsqueeze(-1).expand(B, T, self.kernel, groups, self.group).reshape(B, T, self.kernel, D)
        weight = coeff.float() + base[side].reshape(1, 1, self.kernel, D)
        terms = [weight[:, :, 0] * hidden]
        for tap in range(1, min(self.kernel, T)):  # rows before the block's start read zeros
            shifted = hidden[:, : T - tap].pad((None, (tap, 0), None))
            terms.append(weight[:, :, tap] * shifted)
        return functools.reduce(operator.add, terms)

    # ---- graphs ----

    def inject(self, features: Tensor, start: Any) -> list[UOp]:
        """Stores that write the K/V of target rows: `features` (1, T, len(target_layers) * dim), the target's
        layer inputs at positions start..start + T - 1."""
        self._init_state(features.device)
        B, T, _ = features.shape
        g = self.enc.output_norm(self.fc(features.float()))
        stores = []
        for i, layer in enumerate(self.blk):
            k = layer.attn_k_norm(layer.attn_k(g).reshape(B, T, self.kv_heads, self.head_dim)).transpose(1, 2)
            v = layer.attn_v(g).reshape(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
            stores.append(self._store(i, self._rope(k, start, T), v, start))
        return stores

    def _attention(self, i: int, layer: _Layer, h: Tensor, start: Any) -> Tensor:
        B, T, _ = h.shape
        q = layer.attn_q_norm(layer.attn_q(h).reshape(B, T, self.heads, self.head_dim)).transpose(1, 2)
        k = layer.attn_k_norm(layer.attn_k(h).reshape(B, T, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = layer.attn_v(h).reshape(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, start, T), self._rope(k, start, T)
        cache = Tensor(self.cache[i].uop.after(self._store(i, k, v, start)))
        # keys of positions start + T - span .. start + T - 1: the window before the block's first row, and the block
        first = start + T + self.pad - self.span
        keys = cache.shrink((None, None, None, (first, first + self.span), None)).float()
        positions = Tensor.arange(self.span).reshape(1, -1) + (Tensor(start) + T - self.span)
        rows = Tensor.arange(T).reshape(-1, 1) + Tensor(start)
        valid = (positions >= 0) & (positions > rows - self.window)  # no causal mask inside the block
        mask = valid.where(0.0, -math.inf).reshape(1, 1, T, self.span)
        out = q.float().scaled_dot_product_attention(keys[0], keys[1], attn_mask=mask, enable_gqa=True)
        return layer.attn_output(out.transpose(1, 2).reshape(B, T, self.heads * self.head_dim))

    def hidden(self, model: Any, tokens: Tensor, start: Any) -> Tensor:
        """The block's final hidden (1, T, dim) for `tokens` (1, T): the anchor, then masks, at start..start+T-1."""
        self._init_state(model.token_embd.weight.device)
        x = model.token_embd(tokens).float()
        for i, layer in enumerate(self.blk):
            h = layer.attn_norm(x)
            dynamic = layer.attn_conv_proj(h)
            h = self._conv(h, dynamic, layer.attn_conv_base, 0)
            a = self._conv(self._attention(i, layer, h, start), dynamic, layer.attn_conv_base, 1)
            x = x + a
            h = layer.ffn_norm(x)
            dynamic = layer.ffn_conv_proj(h)
            h = self._conv(h, dynamic, layer.ffn_conv_base, 0)
            f = layer.ffn_down(layer.ffn_gate(h).silu() * layer.ffn_up(h))
            x = x + self._conv(f, dynamic, layer.ffn_conv_base, 1)
        return self.output_norm(x)

    def select(self, hidden: Tensor, logits: Tensor, anchor: Tensor) -> Tensor:
        """DFlash2's greedy walk: `hidden` (T - 1, dim) and `logits` (T - 1, vocab) of block rows 1.., `anchor` (1,)
        the block's first token. Returns the drafts (1, T - 1) int32."""
        rows, k = int(hidden.shape[0]), self.top_k
        picks = [_top(logits[j].float(), k) for j in range(rows)]
        unary = Tensor.stack(*(values for values, _ in picks))  # (rows, k)
        ids = Tensor.stack(*(i for _, i in picks))  # (rows, k) int32
        gate = self.selector_hidden(hidden).float()  # (rows, rank)
        successor = self.selector_successor(ids).float()  # (rows, k, rank)
        drafts, chosen = [], None
        for j in range(rows):
            previous = anchor.reshape(1) if j == 0 else ids[j - 1]
            predecessor = self.selector_predecessor(previous).float()  # (1 or k, rank)
            scores = (predecessor * gate[j]) @ successor[j].T + unary[j]  # (1 or k, k): from predecessor to successor
            row = scores[0] if j == 0 else (Tensor.arange(k).reshape(k, 1) == chosen).where(scores, 0).sum(0)
            chosen = _top(row, 1)[1]  # (1,) index among the candidates
            drafts.append((Tensor.arange(k) == chosen).where(ids[j], 0).sum().reshape(1))
        return Tensor.cat(*drafts).reshape(1, rows).cast(dtypes.int32)
