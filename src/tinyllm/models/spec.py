"""What a model is and what per-sequence state it needs, read from a GGUF header.

The state layout drives slot sizing, checkpoints and the memory plan (docs/02-design.md,
sections 4.4 and 6). Sizes follow tinygrad's state tensors at the pinned commit: an fp16 KV
cache per attention layer, and fp32 recurrent and convolution state per Gated DeltaNet layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinyllm.weights.gguf import GGUFFile

# architectures whose state layout is modelled here
HYBRID_ARCHS = ("qwen35",)  # Gated DeltaNet + attention
ATTENTION_ARCHS = ("llama", "qwen2", "qwen3", "mistral", "gemma")  # plain attention blocks
KV_ITEMSIZE = 2  # fp16
STATE_ITEMSIZE = 4  # fp32


@dataclass(frozen=True)
class StateLayout:
    attention_layers: int
    recurrent_layers: int
    kv_bytes_per_token: int  # all attention layers together
    recurrent_bytes: int  # all recurrent layers together, per slot
    conv_bytes: int  # all convolution windows together, per slot

    @property
    def checkpoint_bytes(self) -> int:
        """What must be copied to resume from a position: everything that is not position-addressed."""
        return self.recurrent_bytes + self.conv_bytes

    def slot_bytes(self, context: int) -> int:
        return self.kv_bytes_per_token * context + self.checkpoint_bytes


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    name: str
    blocks: int  # main blocks, excluding MTP blocks
    mtp_blocks: int
    dim: int
    feed_forward: int
    vocab_size: int
    context_length: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    state: StateLayout | None  # None when the architecture's state is not modelled yet


def _int(g: GGUFFile, key: str, default: int = 0) -> int:
    value = g.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def from_gguf(g: GGUFFile) -> ModelSpec:
    arch = g.architecture
    mtp = _int(g, "nextn_predict_layers")
    blocks = _int(g, "block_count") - mtp
    dim, heads = _int(g, "embedding_length"), _int(g, "attention.head_count")
    kv_heads = _int(g, "attention.head_count_kv", heads)
    head_dim = _int(g, "attention.key_length", dim // heads if heads else 0)
    vocab = g.count("tokenizer.ggml.tokens") if "tokenizer.ggml.tokens" in g.kv else 0

    state: StateLayout | None = None
    kv_per_layer = 2 * kv_heads * head_dim * KV_ITEMSIZE  # keys and values
    if arch in HYBRID_ARCHS:
        interval = _int(g, "full_attention_interval", 1)
        attention = sum(1 for i in range(blocks) if (i + 1) % interval == 0)
        recurrent = blocks - attention
        inner, value_heads = _int(g, "ssm.inner_size"), _int(g, "ssm.time_step_rank")
        key_dim, key_heads, kernel = _int(g, "ssm.state_size"), _int(g, "ssm.group_count"), _int(g, "ssm.conv_kernel")
        value_dim = inner // value_heads if value_heads else 0
        conv_channels = inner + 2 * key_heads * key_dim
        state = StateLayout(
            attention_layers=attention,
            recurrent_layers=recurrent,
            kv_bytes_per_token=attention * kv_per_layer,
            recurrent_bytes=recurrent * value_heads * value_dim * key_dim * STATE_ITEMSIZE,
            conv_bytes=recurrent * (kernel - 1) * conv_channels * STATE_ITEMSIZE,
        )
    elif arch in ATTENTION_ARCHS:
        state = StateLayout(blocks, 0, blocks * kv_per_layer, 0, 0)

    return ModelSpec(
        architecture=arch,
        name=str(g.kv.get("general.name") or g.kv.get("general.basename") or g.path.stem),
        blocks=blocks,
        mtp_blocks=mtp,
        dim=dim,
        feed_forward=_int(g, "feed_forward_length") or _int(g, "expert_feed_forward_length"),
        vocab_size=vocab,
        context_length=_int(g, "context_length"),
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        experts=_int(g, "expert_count"),
        state=state,
    )
