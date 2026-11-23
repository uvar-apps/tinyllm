"""Host-side sampling with OpenAI-level controls (design 4.9, the default path).

A `Sampler` belongs to one request. It owns that request's random generator, so a `seed` reproduces the same
tokens whatever else the engine runs, and the counts its penalties need.

Order, as in vLLM and llama.cpp: logit bias, then repetition / presence / frequency penalties, then greedy
selection if the temperature is 0, else temperature, then the top-k, top-p and min-p filters, then a draw.
Log-probabilities are those of the model's own distribution (log-softmax of the raw logits), unaffected by the
sampling settings, which is what vLLM and llama.cpp report by default.

With a 248k-token vocabulary a full sort costs several milliseconds per token, so the filters work on a partial
selection (`np.argpartition`). Top-k comes first, and top-p and min-p then act on the top-k tokens' renormalized
distribution (as in Hugging Face, vLLM and llama.cpp); without top-k the selection grows until it holds every
token the filters could keep.
"""

from __future__ import annotations

import secrets
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

FIRST_CANDIDATES = 64  # tokens taken by the first partial selection when only top-p / min-p bound the set


@dataclass(frozen=True)
class SamplingParams:
    """What a request asks for. 0 or 1 disable the corresponding filter or penalty."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0  # divides positive logits and multiplies negative ones of tokens already seen
    presence_penalty: float = 0.0  # subtracted once from tokens already generated
    frequency_penalty: float = 0.0  # subtracted per occurrence in the generated tokens
    logit_bias: Mapping[int, float] = field(default_factory=dict)
    seed: int | None = None
    logprobs: bool = False
    top_logprobs: int = 0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be at least 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be at least 0, got {self.top_k}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")
        if not 0 <= self.top_logprobs <= 20:
            raise ValueError(f"top_logprobs must be in [0, 20], got {self.top_logprobs}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    @property
    def device_ok(self) -> bool:
        """Whether a device sampler that sees only the top DEVICE_CANDIDATES logits gives the same result."""
        plain = not self.logit_bias and self.repetition_penalty == 1 and not self.presence_penalty
        plain = plain and not self.frequency_penalty and self.top_logprobs <= DEVICE_TOP_LOGPROBS
        return plain and (self.greedy or 0 < self.top_k <= DEVICE_CANDIDATES)

    @property
    def needs_logits(self) -> bool:
        """False when the most likely token alone decides the result, so an in-graph argmax would do."""
        plain = not self.logit_bias and self.repetition_penalty == 1 and not self.presence_penalty
        return not (self.greedy and plain and not self.frequency_penalty and not self.logprobs)


DEVICE_CANDIDATES = 32  # tinyllm.tg.sampler.CANDIDATES: the layout of a device sampler's result
DEVICE_TOP_LOGPROBS = 20


@dataclass(frozen=True)
class Sample:
    token: int
    logprob: float | None = None  # of `token`, when the request asked for logprobs
    top: tuple[tuple[int, float], ...] = ()  # the `top_logprobs` most likely tokens and their log-probabilities


class Sampler:
    """Samples one request's tokens from full-vocabulary logits."""

    def __init__(self, params: SamplingParams, prompt: Iterable[int] = ()) -> None:
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.seen = set(prompt)  # prompt and generated tokens, for the repetition penalty
        self.generated: Counter[int] = Counter()
        self.count = 0  # tokens generated so far
        # the device sampler's uniform numbers come from (stream, position), not from a running generator
        self.stream = int(params.seed) if params.seed is not None else secrets.randbits(63)

    def accept(self, token: int) -> None:
        """Record a token added to the sequence (sampled here, or decided elsewhere)."""
        self.seen.add(token)
        self.generated[token] += 1
        self.count += 1

    def sample(self, logits: np.ndarray) -> Sample:
        """Pick the next token from one row of logits (shape (vocab,)) and record it."""
        p = self.params
        raw = np.asarray(logits, dtype=np.float32)
        x = self._penalised(raw)
        token = int(np.argmax(x)) if p.greedy else self._draw(x / np.float32(p.temperature))
        result = self._logprobs(raw, token) if p.logprobs or p.top_logprobs else Sample(token)
        self.accept(token)
        return result

    def device_params(self) -> list[float]:
        """What the device sampler needs for the next token: temperature, top_k, top_p, min_p and a uniform
        number (see `device_rows`)."""
        return self.device_rows(1)[0]

    def device_rows(self, count: int) -> list[list[float]]:
        """Device sampler settings for the next `count` generated positions.

        Each position's uniform number depends only on the request's stream (its seed) and the position, so a
        speculative cycle, which samples several positions at once and may discard some, draws the same token at
        each position as one-token decoding does."""
        p = self.params
        if p.greedy:
            return [[0.0, 1.0, 1.0, 0.0, 0.0] for _ in range(count)]
        head = [p.temperature, float(p.top_k), p.top_p, p.min_p]
        return [[*head, float(np.random.default_rng([self.stream, self.count + j]).random())] for j in range(count)]

    def from_device(self, out: np.ndarray) -> Sample:
        """The Sample a device sampler chose; records it. `out`: token, log-sum-exp, candidate logits, their ids."""
        k = DEVICE_CANDIDATES
        token, lse = int(out[0]), float(out[1])
        result = Sample(token)
        if self.params.logprobs or self.params.top_logprobs:
            values, ids = out[2 : 2 + k].astype(np.float64), out[2 + k : 2 + 2 * k].astype(np.int64)
            where = np.flatnonzero(ids == token)
            own = float(values[where[0]]) - lse if where.size else None
            n = self.params.top_logprobs
            result = Sample(
                token, own, tuple((int(i), float(v) - lse) for i, v in zip(ids[:n], values[:n], strict=True))
            )
        self.accept(token)
        return result

    def _penalised(self, raw: np.ndarray) -> np.ndarray:
        p = self.params
        if not (p.logit_bias or p.repetition_penalty != 1 or p.presence_penalty or p.frequency_penalty):
            return raw
        x = raw.copy()
        for token, bias in p.logit_bias.items():
            if 0 <= token < x.size:
                x[token] += np.float32(bias)
        if p.repetition_penalty != 1 and self.seen:
            ids = np.fromiter(self.seen, dtype=np.int64, count=len(self.seen))
            ids = ids[(ids >= 0) & (ids < x.size)]
            values = x[ids]
            x[ids] = np.where(values > 0, values / p.repetition_penalty, values * p.repetition_penalty)
        if (p.presence_penalty or p.frequency_penalty) and self.generated:
            ids = np.fromiter(self.generated.keys(), dtype=np.int64, count=len(self.generated))
            counts = np.fromiter(self.generated.values(), dtype=np.float32, count=len(self.generated))
            keep = (ids >= 0) & (ids < x.size)
            x[ids[keep]] -= np.float32(p.presence_penalty) + np.float32(p.frequency_penalty) * counts[keep]
        return x

    def _draw(self, x: np.ndarray) -> int:
        """Apply top-k, top-p and min-p to scaled logits and draw from what remains."""
        p, n = self.params, x.size
        if 0 < p.top_k < n:
            # top-k first, as Hugging Face, vLLM and llama.cpp do: top-p and min-p then see the top-k tokens'
            # renormalized distribution, and nothing over the whole vocabulary is needed but one partial selection
            ids = np.argpartition(-x, p.top_k - 1)[: p.top_k]
            ids = ids[np.argsort(-x[ids], kind="stable")]
            probs = np.exp(x[ids].astype(np.float64) - float(x[ids[0]]))
            probs /= probs.sum()
            return int(ids[_pick(self._filter(probs), self.rng)])
        top = float(np.max(x))
        weights = np.exp(x - top, dtype=np.float32)
        if p.top_p == 1 and p.min_p == 0:  # no filter: draw from the whole distribution, no sort
            return _pick(weights.astype(np.float64) / float(np.sum(weights, dtype=np.float64)), self.rng)
        limit, k = n, FIRST_CANDIDATES
        log_total = top + float(np.log(np.sum(weights, dtype=np.float64)))
        while True:
            ids = np.argpartition(-x, k - 1)[:k] if k < n else np.arange(n)
            ids = ids[np.argsort(-x[ids], kind="stable")]
            probs = np.exp(x[ids].astype(np.float64) - log_total)
            keep = len(ids)
            if p.top_p < 1:
                # the smallest prefix whose mass reaches top_p, always at least one token
                keep = min(keep, int(np.searchsorted(np.cumsum(probs), p.top_p, side="left")) + 1)
            if p.min_p > 0:
                keep = min(keep, max(1, int(np.sum(probs >= p.min_p * probs[0]))))
            # done when the selection is not the limiting factor: everything allowed, or something filtered out
            if k >= limit or keep < len(ids):
                break
            k = min(limit, 4 * k)
        chosen = probs[:keep] / np.sum(probs[:keep])
        return int(ids[_pick(chosen, self.rng)])

    def _filter(self, probs: np.ndarray) -> np.ndarray:
        """Top-p and min-p over probabilities sorted in descending order; the kept ones, renormalized."""
        p, keep = self.params, probs.size
        if p.top_p < 1:
            keep = min(keep, int(np.searchsorted(np.cumsum(probs), p.top_p, side="left")) + 1)
        if p.min_p > 0:
            keep = min(keep, max(1, int(np.sum(probs >= p.min_p * probs[0]))))
        return probs[:keep] / probs[:keep].sum()

    def _logprobs(self, raw: np.ndarray, token: int) -> Sample:
        top = float(np.max(raw))
        log_total = top + float(np.log(np.sum(np.exp(raw - top, dtype=np.float32), dtype=np.float64)))
        own = float(raw[token]) - log_total
        n = self.params.top_logprobs
        best: tuple[tuple[int, float], ...] = ()
        if n:
            ids = np.argpartition(-raw, n - 1)[:n]
            ids = ids[np.argsort(-raw[ids], kind="stable")]
            best = tuple((int(i), float(raw[i]) - log_total) for i in ids)
        return Sample(token, own, best)


def _pick(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Index drawn from `probs` (sums to 1) with one uniform number."""
    index = int(np.searchsorted(np.cumsum(probs), rng.random(), side="right"))
    return min(index, probs.size - 1)
