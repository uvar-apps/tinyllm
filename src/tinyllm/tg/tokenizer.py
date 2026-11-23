"""tinygrad's GGUF tokenizer (tinygrad.llm.tokenizer.SimpleTokenizer) as a tinyllm.frontend.tokenizer.Tokenizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tinygrad.llm.tokenizer import SimpleTokenizer

# Qwen's generation config ends a turn on either token; the GGUF names only one of them as eos
EXTRA_END_TOKENS = ("<|im_end|>", "<|endoftext|>")


class GGUFTokenizer:
    def __init__(self, kv: Mapping[str, Any]) -> None:
        """`kv`: the GGUF metadata with the vocabulary arrays read in full."""
        self._tok = SimpleTokenizer.from_gguf_kv(dict(kv))
        tok = self._tok
        self.special_tokens: Mapping[str, int] = dict(tok._special_tokens)
        self.chat_template: str | None = tok.chat_template
        self.bos_token = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
        self.eos_token = tok.decode([tok.eos_id])
        ends = {tok.eos_id, tok.eot_id, tok.eom_id} | {tok._special_tokens.get(t) for t in EXTRA_END_TOKENS}
        self.end_tokens = frozenset(t for t in ends if t is not None)
        self.vocab_size = len(tok._tok2bytes)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text))

    def decode(self, tokens: list[int]) -> str:
        return str(self._tok.decode(tokens))

    def token_bytes(self, token: int) -> bytes:
        return bytes(self._tok._tok2bytes[token])

    def is_end(self, token: int) -> bool:
        return token in self.end_tokens
