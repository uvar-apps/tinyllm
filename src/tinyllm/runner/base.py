"""The interface between the engine and whatever runs the model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Runner(Protocol):
    """One sequence's model state, advanced token by token (phase 1; phase 2 adds rows)."""

    max_context: int
    vocab_size: int
    device_sampling: bool  # whether `feed_sampled` is available

    def begin(self, tokens: Sequence[int]) -> int:
        """Prepare to run `tokens` (a whole prompt) and return how many leading tokens are already in the state.

        At least one token is always left to feed, since the logits after the last prompt token are needed.
        """
        ...

    def feed(self, tokens: Sequence[int]) -> np.ndarray:
        """Run `tokens` after what the state already holds; the logits after the last one, shape (vocab,).

        The array may be reused by the next call: use it before calling again.
        """
        ...

    def feed_sampled(self, token: int, params: Sequence[float]) -> np.ndarray:
        """Run one token and sample the next on the device, when `device_sampling` is true (see tinyllm.tg.sampler
        for the layout of `params` and of the result)."""
        ...

    def warmup(self) -> None:
        """Compile every program a request can need, so the first request does not pay for it."""
        ...
