"""`Runner` over tinygrad.llm's model (`tinyllm.tg.upstream.UpstreamSession`), one sequence at a time."""

from __future__ import annotations

import functools
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tinyllm.runner.checkpoints import CHECKPOINT_EVERY, CHECKPOINTS, Checkpoints, common_prefix
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
        drafter: str | Path | None = None,
        checkpoints: int = CHECKPOINTS,
        checkpoint_every: int = CHECKPOINT_EVERY,
    ) -> None:
        self.session = UpstreamSession(
            model_path,
            max_context,
            rows=1,
            chunk=chunk,
            logits=True,
            speculative=speculative,
            drafter=drafter,
            checkpoints=checkpoints,
        )
        # where a new prompt that departs from the held tokens can resume (0 slots: only exact extensions)
        self.checkpoints = Checkpoints(self.session.checkpoints)
        self.checkpoint_every = checkpoint_every
        self._prompt_end = 0  # the length of the prompt `begin` was given: feeds up to it are prompt, past it decoding
        # drafts per cycle, from the MTP block (`drafter` None) or a DFlash2 GGUF; 0 without either
        self.speculative = self.session.speculative
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
        # The recurrent (DeltaNet) state is only valid at the exact position it was taken. A sequence that extends
        # what the state holds continues from there; one that departs from it resumes from the newest checkpoint
        # below the point of departure (tinyllm.runner.checkpoints); otherwise the model starts over.
        self._prompt_end = len(tokens)
        held = len(self.tokens)
        shared = common_prefix(self.tokens, tokens)
        if 0 < held < len(tokens) and shared == held:
            return held
        best = self.checkpoints.best(min(shared, len(tokens) - 1))  # at least one token is left to feed
        if best is not None:
            self.session.load_state(best.slot, best.pos)
            del self.tokens[best.pos :]
            self.checkpoints.truncate(best.pos)
            return best.pos
        self.session.reset()
        self.tokens = []
        self.checkpoints.clear()
        return 0

    def _checkpoint(self, prompt: bool) -> None:
        """Copy the state at the session's position into a checkpoint slot."""
        slot = self.checkpoints.add(self.session.pos, prompt)
        if slot is not None:
            self.session.save_state(slot)

    def _after_decode(self) -> None:
        if len(self.session.trace) > 2 * TRACE_KEEP:
            del self.session.trace[:-TRACE_KEEP]
        if self.checkpoints.slots and len(self.tokens) - self.checkpoints.newest >= self.checkpoint_every:
            self._checkpoint(prompt=False)

    def feed(self, tokens: Sequence[int], following: int | None = None) -> np.ndarray:
        """The logits after the last token; the array is overwritten by the next call. `following`: the token that
        comes next when the caller already knows it (the rest of a prompt), for the MTP block's KV."""
        ids = [int(t) for t in tokens]
        if not ids:
            raise ValueError("feed needs at least one token")
        prompt = len(self.tokens) < self._prompt_end
        # the prompt's last slice: a checkpoint before its last prefill call, which the next request of an agent
        # loop resumes from when the chat template renders this request's reply differently from how it came out
        last = self.checkpoints.slots and prompt and len(self.tokens) + len(ids) == self._prompt_end
        # one token runs the decode program (prefill's would compute a whole padded chunk); with an MTP block it
        # leaves the MTP's KV for that row unwritten, which only makes the next first draft less likely to hold
        if len(ids) == 1:
            if last:
                self._checkpoint(prompt=True)
            logits = self.session.step(ids)
        else:
            hook = functools.partial(self._checkpoint, True) if last else None
            logits = self.session.prefill(ids, wait_each=False, following=following, before_last=hook)
        self.tokens.extend(ids)
        if prompt:
            if len(self.session.trace) > 2 * TRACE_KEEP:
                del self.session.trace[:-TRACE_KEEP]
        else:
            self._after_decode()
        return np.asarray(logits[0])

    def feed_sampled(self, token: int, params: Sequence[float]) -> np.ndarray:
        """Run one token and sample the next on the GPU: the packed result of `tinyllm.tg.sampler.sample_row`.

        `params`: temperature, top_k, top_p, min_p, u (a uniform number in [0, 1)). The array is reused by the
        next call."""
        out = self.session.decode_sampled([int(token)], params)
        self.tokens.append(int(token))
        self._after_decode()
        return np.asarray(out)

    def feed_speculative(self, token: int, params: Sequence[Sequence[float]]) -> tuple[np.ndarray, int]:
        """One draft-and-verify cycle after `token` (tinyllm.tg.speculative): (K + 1, OUTPUT) device-sampler
        results and n; rows 0..n are the n + 1 tokens produced. `params`: one row of settings per verify row.
        The array is reused by the next call."""
        rows, n = self.session.speculate(int(token), params)
        self.tokens.append(int(token))
        self.tokens.extend(int(t) for t in rows[:n, 0])
        self._after_decode()
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
        if self.checkpoints.slots:  # the second call captures each program, the third replays it with another slot
            other = min(1, self.checkpoints.slots - 1)
            for slot in (0, other, 0):
                self.session.save_state(slot)
            for slot in (0, other, 0):
                self.session.load_state(slot, self.session.pos)
        self.session.reset()
        self.tokens = []
        self.checkpoints.clear()
        self.session.trace.clear()
        self.warmup_seconds = time.perf_counter() - start
