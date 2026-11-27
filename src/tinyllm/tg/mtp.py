"""Qwen3.8's multi-token prediction (MTP) block, the draft model for speculative decoding (plan, phase 3).

Qwen3.8 GGUFs carry one extra block after the main layers (`<arch>.nextn_predict_layers` = 1, `blk.64` on the
27B) that tinygrad.llm leaves unloaded. It is one gated-attention layer of the main architecture with its own KV
cache and three extra norms and a projection, and it shares the main model's token embedding and output head
(llama.cpp, `src/models/qwen35.cpp`, `graph_mtp`):

    x = eh_proj(concat(enorm(embed(token)), hnorm(h)))      # h: the main model's output-normed hidden
    x = attention block (attn_norm, gated attention with its own KV cache, residual, ffn_norm, SiLU FFN, residual)
    g = shared_head_norm(x)                                  # the next draft step's h
    logits = output(g)                                       # the main model's head

Row i of the MTP sequence pairs the main model's hidden at position i with the token at position i + 1, and
predicts the token at i + 2. RoPE sees index i; attention depends only on position differences, so this matches
llama.cpp's convention of index i + 1.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from tinygrad import Device, Tensor, dtypes, nn
from tinygrad.llm.gguf import gguf_parse, gguf_shard
from tinygrad.llm.kernels.amd import Linear
from tinygrad.llm.model import GatedResidual, GroupedRMSNorm, Transformer, TransformerBlock


class _NextN:
    def __init__(self, dim: int, eps: float) -> None:
        self.eh_proj = Linear(2 * dim, dim, bias=False)
        self.enorm, self.hnorm, self.shared_head_norm = (nn.RMSNorm(dim, eps) for _ in range(3))


class MTPBlock(TransformerBlock):  # type: ignore[misc]
    """The nextn block: a main-model attention layer plus the projection that feeds it."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.nextn = _NextN(config.dim, config.norm_eps)

    def hidden(self, model: Any, tokens: Tensor, h: Tensor, start_pos: tuple[Any, ...]) -> Tensor:
        """The block's output after `shared_head_norm` for `tokens` (B, T) paired with main hiddens `h` (B, T, D)."""
        e = model.token_embd(tokens.to(model.token_embd.weight.device)).float()
        x = self.nextn.eh_proj(self.nextn.enorm(e).cat(self.nextn.hnorm(h.float()), dim=-1))
        x, _ = self(x, start_pos, None, tokens)
        return self.nextn.shared_head_norm(x)

    def head(self, x: Tensor) -> Tensor:
        """What the output head reads: `hidden`'s result as it is."""
        return x


def quantize_q4_0(weight: Tensor, step: int | None = None) -> Tensor:
    """`weight` (out, in) as Q4_0 blocks, as the uint16 halfwords tinyllm's Q4_0 kernel reads (ggml's reference
    rounding: d = the signed value of largest magnitude / -8, code = min(15, trunc(x / d + 8.5))). `step` rows at a
    time, all at once by default: one pass over the 27B's head takes about a second, sixteen take 15 (scheduling)."""
    out_features, in_features = weight.shape
    assert in_features % 32 == 0
    step = step or out_features
    parts = []
    for lo in range(0, out_features, step):
        x = weight[lo : min(lo + step, out_features)].float().reshape(-1, 32)
        top, bottom = x.max(-1), x.min(-1)
        d = (top.abs() >= bottom.abs()).where(top, bottom) / -8
        inv = (d != 0).where(1 / d, 0)
        q = (x * inv.unsqueeze(-1) + 8.5).floor().clip(0, 15).cast(dtypes.uint8)
        nibbles = q[:, :16] | (q[:, 16:] << 4)
        scale = d.cast(dtypes.float16).bitcast(dtypes.uint8).reshape(-1, 2)
        parts.append(scale.cat(nibbles, dim=-1).flatten().contiguous().realize())
    return Tensor.cat(*parts).bitcast(dtypes.uint16).contiguous().realize()


def draft_head(output: Any, rows: int | None = None) -> Any:
    """A cheaper copy of the output head for drafting: Q4_0, and only the first `rows` tokens when given (token
    ids in a BPE vocabulary roughly follow frequency; a draft of a token outside them is simply rejected).

    Drafting reads the whole head once per draft, 1 GB in Q6_K on the 27B's UD-Q4_K_XL and 2.5 GB in BF16 on its
    UD-Q8_K_XL; the verify still uses the real head, so this changes acceptance, never the output."""
    from tinyllm.tg.kernels import q4

    weight = output.weight if rows is None else output.weight[:rows]
    head = Linear(int(output.in_features), int(weight.shape[0]), bias=False)
    head.weight = quantize_q4_0(weight)
    head._tinyllm_q4 = (q4.GGML_Q4_0, head.weight)
    head._tinyllm_kernel = "q4"  # tinyllm.tg.kernels.hook: run it with the Q4_0 kernel
    return head


@contextlib.contextmanager
def _capture_unused(store: dict[str, Tensor]) -> Iterator[None]:
    """While loading a model, keep the state-dict entries the model has no place for (the MTP block's)."""
    original = nn.state.load_state_dict

    def load_state_dict(model: Any, state_dict: dict[str, Tensor], *args: Any, **kwargs: Any) -> Any:
        ret = original(model, state_dict, *args, **kwargs)
        if kwargs.get("consume"):
            store.update(state_dict)  # what is left after the model took its own entries
        return ret

    nn.state.load_state_dict = load_state_dict
    try:
        yield
    finally:
        nn.state.load_state_dict = original


def load(path: str, max_context: int | None, realize: bool) -> tuple[Transformer, dict[str, Any], MTPBlock | None]:
    """`Transformer.from_gguf` plus the MTP block, or None for a GGUF without one."""
    unused: dict[str, Tensor] = {}
    with _capture_unused(unused):
        model, kv = Transformer.from_gguf(path, max_context, realize=realize)
    arch = kv["general.architecture"]
    if int(kv.get(f"{arch}.nextn_predict_layers", 0)) != 1:
        return model, kv, None
    prefix = f"blk.{model.config.num_blocks}."
    weights = {k[len(prefix) :]: v for k, v in unused.items() if k.startswith(prefix)}
    block = MTPBlock(model.config)
    nn.state.load_state_dict(block, weights, verbose=False, consume=True, realize=False)
    if realize:
        params = nn.state.get_parameters(block)
        for p in params:
            p.replace(p.contiguous())
        Tensor.realize(*params)
    return model, kv, block


class _StreamNextN:
    """`nextn` of a model with residual streams: the projection (applied per stream) and the head's mixer."""

    def __init__(self, config: Any) -> None:
        dim, count = config.dim, config.hc.count
        self.eh_proj = Linear(2 * dim, dim, bias=False)
        self.enorm = nn.RMSNorm(dim, config.norm_eps)
        self.hnorm = GroupedRMSNorm(count, dim, config.norm_eps)
        self.hc_head = GatedResidual(config, inject=False)


class StreamMTPBlock(TransformerBlock):  # type: ignore[misc]
    """Qwen3.8-Flash-Next's MTP block (llama.cpp b12675, `src/models/qwen4exp.cpp`, `graph_mtp`): one sparse-attention
    layer of the main architecture, with its own experts, caches and residual mixers, fed per stream by

        x_s = eh_proj(concat(enorm(embed(token)), hnorm(h)_s))    # h: the main model's streams before its final mixer

    Its streams are the next draft step's h; the token comes from the main model's head over `head(x)`, the block's
    own final mixer. The block ships in a GGUF of its own (`load_streams`)."""

    def __init__(self, config: Any) -> None:
        super().__init__(replace(config, ple=None))
        self.nextn = _StreamNextN(config)

    def hidden(self, model: Any, tokens: Tensor, h: Tensor, start_pos: tuple[Any, ...]) -> Tensor:
        """The block's streams (B, T, count, D) for `tokens` (B, T) paired with the main model's streams `h`."""
        B, T, count, dim = h.shape
        e = model.token_embd(tokens.to(model.token_embd.weight.device)).float()
        en = self.nextn.enorm(e).unsqueeze(2).expand(B, T, count, dim)
        x = en.cat(self.nextn.hnorm(h.float()), dim=-1).reshape(B, T * count, 2 * dim)
        x = self.nextn.eh_proj(x).reshape(B, T, count, dim)
        x, _ = self(x, start_pos, None, tokens)
        return x

    def head(self, x: Tensor) -> Tensor:
        """What the output head reads: the block's streams (B, T, count, D) mixed into one (B, T, D)."""
        return self.nextn.hc_head.mix(x)[0]


def load_streams(model: Any, path: str, realize: bool) -> StreamMTPBlock:
    """The MTP block of a model with residual streams from its own GGUF (`mtp-*.gguf`: one block after the main
    model's, with or without the embedding and head it shares with the main model)."""
    kv, entries = gguf_parse(path)
    arch = kv["general.architecture"]
    if int(kv.get(f"{arch}.nextn_predict_layers", 0)) != 1:
        raise ValueError(f"{path} carries no MTP block")
    prefix = f"blk.{model.config.num_blocks}."
    state = gguf_shard({k: v for k, v in entries.items() if k.startswith(prefix)}, (Device.DEFAULT,))
    weights = {}
    for key, value in state.items():  # the names `Transformer.from_gguf` gives a block's tensors (pinned commit)
        name = re.sub(r"^(hc_attn|hc_ffn)_", r"\1.", key[len(prefix) :])
        name = re.sub(r"^nextn\.hc_head_", "nextn.hc_head.", name)
        name = name.replace("indexer.q_proj", "indexer.attn_q_b").replace("indexer.k_proj", "indexer.attn_k")
        weights[name] = value.cast("float16") if dtypes.is_float(value.dtype) else value
    if not weights:
        raise ValueError(f"{path} has no tensors of block {model.config.num_blocks}")
    block = StreamMTPBlock(model.config)
    nn.state.load_state_dict(block, weights, verbose=False, consume=True, realize=False)
    if weights:
        raise ValueError(f"{path}: tensors the MTP block has no place for: {sorted(weights)[:5]}")
    # the mixers' inject weights (count x count * dim) as float32, as the main model's file stores its own: the
    # mixer kernel reads them as a matrix (tinyllm.tg.kernels.hc), which a quantized layer's packed words are not
    for mixer in (block.hc_attn, block.hc_ffn):
        mixer.inject.weight = mixer.inject.weight.float().contiguous().realize()
    if realize:
        params = nn.state.get_parameters(block)
        for p in params:
            p.replace(p.contiguous())
        Tensor.realize(*params)
    return block
