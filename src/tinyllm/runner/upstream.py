"""`Runner` over tinygrad.llm's model (`tinyllm.tg.upstream.UpstreamSession`), one sequence at a time."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tinyllm.tg import default_device, set_wait
from tinyllm.tg.upstream import StepRecord, UpstreamSession

DEFAULT_CHUNK = 128  # prompt tokens per prefill call (plan, phase 1 item 7)
TRACE_KEEP = 4096  # step records kept; a server runs for days and must not grow


class UpstreamRunner:
    def __init__(
        self,
        model_path: str | Path,
        max_context: int = 32768,
        chunk: int = DEFAULT_CHUNK,
        wait: str = "yield",
        speculative: int = 0,
    ) -> None:
        self.session = UpstreamSession(
            model_path, max_context, rows=1, chunk=chunk, logits=True, speculative=speculative
        )
        self.speculative = self.session.speculative  # drafts per cycle; 0 without an MTP block
        set_wait(wait)  # the HTTP thread must get the interpreter while the GPU works (tinyllm.tg.set_wait)
        self.session.reuse_logits = True  # `feed`'s result is used before the next call
        # sampling inside the decode program needs the AMD top-k kernel and memory the host can map
        self.device_sampling = default_device().startswith("AMD")
        self.chunk = chunk
        self.max_context = self.session.max_context
        self.vocab_size = self.session.vocab_size
        self.kv: dict[str, Any] = self.session.kv  # the GGUF metadata, for the tokenizer and the chat template
        self.tokens: list[int] = []  # what the model state holds, in order
        self.warmup_seconds = 0.0

    @property
    def trace(self) -> list[StepRecord]:
        return self.session.trace

    def begin(self, tokens: Sequence[int]) -> int:
        # The recurrent (DeltaNet) state is only valid at the exact position it was taken, so the state is reused
        # only when the new sequence extends what it holds; otherwise the model starts over.
        held = len(self.tokens)
        if 0 < held < len(tokens) and list(tokens[:held]) == self.tokens:
            return held
        self.session.reset()
        self.tokens = []
        return 0

    def feed(self, tokens: Sequence[int], following: int | None = None) -> np.ndarray:
        """The logits after the last token; the array is overwritten by the next call. `following`: the token that
        comes next when the caller already knows it (the rest of a prompt), for the MTP block's KV."""
        ids = [int(t) for t in tokens]
        if not ids:
            raise ValueError("feed needs at least one token")
        # one token runs the decode program (prefill's would compute a whole padded chunk); with an MTP block it
        # leaves the MTP's KV for that row unwritten, which only makes the next first draft less likely to hold
        if len(ids) == 1:
            logits = self.session.step(ids)
        else:
            logits = self.session.prefill(ids, wait_each=False, following=following)
        self.tokens.extend(ids)
        if len(self.session.trace) > 2 * TRACE_KEEP:
            del self.session.trace[:-TRACE_KEEP]
        return np.asarray(logits[0])

    def feed_sampled(self, token: int, params: Sequence[float]) -> np.ndarray:
        """Run one token and sample the next on the GPU: the packed result of `tinyllm.tg.sampler.sample_row`.

        `params`: temperature, top_k, top_p, min_p, u (a uniform number in [0, 1)). The array is reused by the
        next call."""
        out = self.session.decode_sampled([int(token)], params)
        self.tokens.append(int(token))
        if len(self.session.trace) > 2 * TRACE_KEEP:
            del self.session.trace[:-TRACE_KEEP]
        return np.asarray(out)

    def feed_speculative(self, token: int, params: Sequence[Sequence[float]]) -> tuple[np.ndarray, int]:
        """One draft-and-verify cycle after `token` (tinyllm.tg.speculative): (K + 1, OUTPUT) device-sampler
        results and n; rows 0..n are the n + 1 tokens produced. `params`: one row of settings per verify row.
        The array is reused by the next call."""
        rows, n = self.session.speculate(int(token), params)
        self.tokens.append(int(token))
        self.tokens.extend(int(t) for t in rows[:n, 0])
        if len(self.session.trace) > 2 * TRACE_KEEP:
            del self.session.trace[:-TRACE_KEEP]
        return rows, n

    def warmup(self) -> None:
        """Compile the full-chunk and partial-chunk prefill programs, both decode programs (logits out, and sampled
        on the GPU) and, with speculative decoding, the cycle and its K restore programs (each runs at least twice,
        the second run captures it), then forget the state."""
        start = time.perf_counter()
        for _ in range(2):
            self.begin([0])
            self.feed([0] * (self.chunk + 3))  # one full chunk, then a partial one
            self.feed([0])
            self.feed([0])
            if self.device_sampling:
                self.feed_sampled(0, [1.0, 20, 0.95, 0.0, 0.5])
                self.feed_sampled(0, [1.0, 20, 0.95, 0.0, 0.5])
            if self.speculative:  # draft-and-verify cycles, and the restore program for each rejected row
                rows = [[1.0, 20, 0.95, 0.0, 0.5]] * (self.speculative + 1)
                for _ in range(3):
                    self.feed_speculative(0, rows)
                for n in range(self.speculative):
                    self.session._restore_row = n
                    self.session._settle()
        self.session.reset()
        self.tokens = []
        self.session.trace.clear()
        self.warmup_seconds = time.perf_counter() - start
