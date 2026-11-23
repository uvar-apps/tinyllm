"""Test doubles: a runner whose logits are predictable, and a tokenizer for it."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np


class CountingRunner:
    """The most likely next token is always the last token + 1 (mod vocab), so generation counts upward."""

    def __init__(
        self, vocab_size: int = 300, max_context: int = 1 << 30, fail_on: int | None = None, delay: float = 0.0
    ) -> None:
        self.vocab_size, self.max_context, self.fail_on, self.delay = vocab_size, max_context, fail_on, delay
        self.tokens: list[int] = []
        self.calls: list[int] = []  # tokens per feed call
        self.device_sampling = False

    def begin(self, tokens: Sequence[int]) -> int:
        held = len(self.tokens)
        if 0 < held < len(tokens) and list(tokens[:held]) == self.tokens:
            return held
        self.tokens = []
        return 0

    def feed(self, tokens: Sequence[int]) -> np.ndarray:
        if self.fail_on is not None and self.fail_on in tokens:
            raise RuntimeError("device fault")
        if self.delay:
            time.sleep(self.delay)
        self.tokens.extend(int(t) for t in tokens)
        self.calls.append(len(tokens))
        logits = (-10.0 - 1e-3 * np.arange(self.vocab_size)).astype(np.float32)  # distinct values: no ties
        logits[(self.tokens[-1] + 1) % self.vocab_size] = 10.0
        return logits

    def warmup(self) -> None:
        pass


class DeviceCountingRunner(CountingRunner):
    """CountingRunner with a device sampler, modelled in NumPy on tinyllm.tg.sampler.sample_row."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.sampled_calls = 0
        self.device_sampling = True

    def feed_sampled(self, token: int, params: Sequence[float]) -> np.ndarray:
        self.sampled_calls += 1
        x = self.feed([token]).astype(np.float64)
        k = 32
        order = np.lexsort((np.arange(x.size), -x))[:k]
        vals = x[order]
        temperature, top_k, top_p, min_p, u = params
        p = np.exp((vals - vals[0]) / max(temperature, 1e-6)) * (np.arange(k) < top_k)
        p /= p.sum()
        q = p * (((np.cumsum(p) - p) < top_p) & (p >= min_p * p[0]))
        q /= q.sum()
        choice = min(int(np.sum(np.cumsum(q) <= u)), k - 1)
        lse = float(np.log(np.sum(np.exp(x - x.max()))) + x.max())
        return np.concatenate([[order[choice], lse], vals, order]).astype(np.float32)


class ByteTokenizer:
    """Token ids 0..255 are bytes; 256.. are control tokens. Matches CountingRunner's vocabulary of 300."""

    def __init__(self) -> None:
        self.special_tokens = {"<|im_start|>": 256, "<|im_end|>": 257, "<think>": 258, "</think>": 259}
        self.chat_template = (
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and "
            "enable_thinking is false %}<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}"
        )
        self.bos_token, self.eos_token = "", "<|im_end|>"
        self.end_tokens = frozenset({257})

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for name, token in self.special_tokens.items():
                if text.startswith(name, i):
                    out.append(token)
                    i += len(name)
                    break
            else:
                out.extend(text[i].encode())
                i += 1
        return out

    def token_bytes(self, token: int) -> bytes:
        names = {v: k for k, v in self.special_tokens.items()}
        return names[token].encode() if token in names else bytes([token]) if token < 256 else b""

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.token_bytes(t) for t in tokens).decode(errors="replace")

    def is_end(self, token: int) -> bool:
        return token == 257


class ScriptRunner:
    """Emits a fixed token script, one token per step, whatever the prompt; logits optionally noisy."""

    def __init__(
        self, script: Sequence[int], vocab_size: int = 300, max_context: int = 1 << 20, noise: float = 0.0
    ) -> None:
        self.script, self.vocab_size, self.max_context, self.noise = list(script), vocab_size, max_context, noise
        self.tokens: list[int] = []
        self.step = 0
        self.prompts: list[list[int]] = []
        self.device_sampling = False
        self.hold: object | None = None  # a threading.Event: feed waits for it (to keep a request running)

    def begin(self, tokens: Sequence[int]) -> int:
        self.prompts.append(list(tokens))
        self.tokens, self.step = [], 0
        return 0

    def feed(self, tokens: Sequence[int]) -> np.ndarray:
        if self.hold is not None:
            self.hold.wait()  # type: ignore[attr-defined]
        self.tokens.extend(int(t) for t in tokens)
        logits = np.full(self.vocab_size, -30.0, dtype=np.float32)
        if self.noise:
            logits = (
                np.random.default_rng(len(self.tokens)).standard_normal(self.vocab_size).astype(np.float32) * self.noise
            )
        logits[self.script[min(self.step, len(self.script) - 1)]] += 40.0 if not self.noise else 0.0
        self.step += 1
        return logits

    def warmup(self) -> None:
        pass
