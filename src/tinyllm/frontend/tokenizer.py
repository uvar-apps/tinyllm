"""The tokenizer as the frontend sees it. `tinyllm.tg.tokenizer` adapts tinygrad's GGUF tokenizer to it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class Tokenizer(Protocol):
    special_tokens: Mapping[str, int]  # text of each control token -> id
    chat_template: str | None  # the Jinja chat template stored with the model, if any
    bos_token: str  # "" when the model has none
    eos_token: str
    end_tokens: frozenset[int]  # ids that end a turn when generated

    def encode(self, text: str) -> list[int]:
        """Token ids for `text`; control tokens written in the text become their ids."""
        ...

    def token_bytes(self, token: int) -> bytes:
        """The bytes one token stands for (a token can hold part of a UTF-8 character)."""
        ...

    def is_end(self, token: int) -> bool:
        """Whether generating `token` ends the turn."""
        ...

    def decode(self, tokens: list[int]) -> str: ...
