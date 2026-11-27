"""`Runner` over tinygrad.llm's model (`tinyllm.tg.upstream.UpstreamSession`), one sequence at a time."""

from __future__ import annotations

import functools
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tinyllm.runner.checkpoints import (
    CHECKPOINT_EVERY,
    CHECKPOINTS,
    Checkpoint,
    Checkpoints,
    Ownership,
    common_prefix,
    plan,
)
from tinyllm.tg import default_device, set_wait, synchronize
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
        mtp_path: str | Path | None = None,
        copy: int = 0,
        programs: str | Path | None = None,
    ) -> None:
        # `programs`: a directory the compiled programs are kept in between starts (tinyllm.tg.programs)
        self.session = UpstreamSession(
            model_path,
            max_context,
            rows=1,
            chunk=chunk,
            logits=True,
            speculative=speculative,
            drafter=drafter,
            checkpoints=checkpoints,
            mtp_path=mtp_path,
            copy=copy,
            programs=programs,
        )
        # where a new prompt that departs from the held tokens can resume (0 slots: only exact extensions)
        self.checkpoints = Checkpoints(self.session.checkpoints)
        self.checkpoint_every = checkpoint_every
        self._prompt_end = 0  # the length of the prompt `begin` was given: feeds up to it are prompt, past it decoding
        # drafts per cycle, from the MTP block (`drafter` None; the model's own or `mtp_path`'s) or a DFlash2 GGUF;
        # 0 without either
        self.speculative = self.session.speculative
        # rows of sampler settings a cycle takes, and drafts of a block copied from earlier text (0: none)
        self.speculative_rows = int(getattr(self.session, "speculative_rows", self.speculative + 1))
        self.copy = int(getattr(self.session, "copy", 0)) if self.speculative else 0
        set_wait(wait)  # the HTTP thread must get the interpreter while the GPU works (tinyllm.tg.set_wait)
        self.session.reuse_logits = True  # `feed`'s result is used before the next call
        # sampling inside the decode program needs the AMD top-k kernel and memory the host can map
        self.device_sampling = default_device().startswith("AMD")
        self.chunk = chunk
        self.max_context = self.session.max_context
        self.vocab_size = self.session.vocab_size
        self.kv: dict[str, Any] = self.session.kv  # the GGUF metadata, for the tokenizer and the chat template
        self.tokens: list[int] = []  # what the model state holds, in order
        self.reused = 0  # tokens of the prompt `begin` was given whose state is reused (the rest is run)
        self._prompt: list[int] | None = None  # that prompt
        self._fed = 0  # how many of its tokens `feed` has been handed
        self._jumps: list[tuple[int, Checkpoint]] = []  # (at, checkpoint): once `at` tokens are held, load it
        self._owned = Ownership()  # which sequence the caches of each position were computed from
        self._live: list[list[int]] = []  # [first, end) ranges of positions written with `tokens` since then
        self._epoch = 0  # counts requests, for the checkpoints' eviction order
        self._last_checkpoint = 0  # the newest checkpoint of the held sequence (0: none)
        # one past the last position written with entries of no sequence since the held one's were last recorded:
        # the padding rows of a padded prefill call, the rejected rows of a speculative verify
        self._written_end = 0
        self.warmup_seconds = 0.0
        self.warmup_stages: dict[str, float] = {}  # seconds `warmup` spent by program

    @property
    def trace(self) -> list[StepRecord]:
        return self.session.trace

    def begin(self, tokens: Sequence[int]) -> int:
        """Prepare to run the prompt `tokens` and return the position `feed` is to be handed it from.

        The recurrent (DeltaNet) state is only valid at the exact position it was taken. A prompt that extends
        what the state holds continues from there. Any other resumes from saved states of its own first tokens
        (tinyllm.runner.checkpoints): `plan` picks the checkpoint to start from, and where a later one lets the
        state jump ahead once the cache positions another request overwrote have been run again. `feed` makes
        those jumps; `reused` is how many of the prompt's tokens are not run."""
        prompt = [int(t) for t in tokens]
        n = len(prompt)
        self._prompt, self._prompt_end, self._jumps = prompt, n, []
        self._epoch += 1
        held = len(self.tokens)
        if 0 < held < n and common_prefix(self.tokens, prompt) == held:
            self._fed = self.reused = held
            return held
        self._settle()
        ids = np.asarray(prompt, dtype=np.int64)
        found = {c.pos: c for c in self.checkpoints.matching(ids, n - 1)}
        self.checkpoints.touch(list(found.values()), self._epoch)
        start, jumps, run = plan(self._owned.valid(ids), sorted(found))
        if start:
            self._load(found[start])
        else:
            self.session.reset()
            self.tokens, self._last_checkpoint = [], 0
        self._jumps = [(at, found[pos]) for at, pos in jumps]
        self._fed, self.reused = start, n - run
        return start

    def forget(self) -> None:
        """Drop the state, every checkpoint and the record of what the caches hold: the next prompt starts over."""
        self.session.reset()
        self.tokens, self._prompt, self._jumps, self._live, self._last_checkpoint = [], None, [], [], 0
        self._written_end = 0
        self.checkpoints.clear()
        self._owned.clear()

    def _settle(self) -> None:
        """Record the cache positions written with the held sequence, before the state leaves it."""
        if self._live:
            owner = np.asarray(self.tokens, dtype=np.int64)
            for i, (first, end) in enumerate(self._live):
                # past the end of what was kept lie a verify's rejected rows and a padded call's padding rows
                spill = max(self._written_end - end, 0) if i == len(self._live) - 1 else 0
                self._owned.wrote(first, end, owner, spill)
            self._live, self._written_end = [], 0

    def _load(self, checkpoint: Checkpoint) -> None:
        """Put the state at a checkpoint of the current prompt."""
        assert self._prompt is not None
        self._settle()
        self.session.load_state(checkpoint.slot, checkpoint.pos, self._prompt[: checkpoint.pos])
        self.tokens = self._prompt[: checkpoint.pos]
        self._last_checkpoint = checkpoint.pos

    def _extend(self, ids: Sequence[int]) -> None:
        """The state now holds `ids` too: their positions' caches are the held sequence's."""
        first = len(self.tokens)
        self.tokens.extend(ids)
        if self._live and self._live[-1][1] == first:
            self._live[-1][1] = len(self.tokens)
        else:
            self._live.append([first, len(self.tokens)])

    def _checkpoint(self, prompt: bool, inside: int | None = None) -> None:
        """Copy the state at the session's position into a checkpoint slot, or with `inside` the state a padded
        prefill call kept at that position."""
        pos = self.session.pos if inside is None else inside
        # before a prompt's last call the held list has not been extended to `pos` yet
        source = self._prompt if prompt and self._prompt is not None else self.tokens
        if len(source) < pos:
            return
        slot = self.checkpoints.add(pos, prompt, source, self._epoch)
        if slot is not None:
            if inside is None:
                self.session.save_state(slot)
            else:
                self.session.save_mid_state(slot)
        self._last_checkpoint = pos

    def _after_decode(self) -> None:
        if len(self.session.trace) > 2 * TRACE_KEEP:
            del self.session.trace[:-TRACE_KEEP]
        if self.checkpoints.slots and len(self.tokens) - self._last_checkpoint >= self.checkpoint_every:
            self._checkpoint(prompt=False)

    def _run_prompt(self, ids: list[int], following: int | None, final: bool) -> np.ndarray:
        """Run a stretch of the prompt after what the state holds; `final`: it ends the prompt."""
        # the prompt's last stretch: a checkpoint before its last prefill call, which the next request of an agent
        # loop resumes from when the chat template renders this request's reply differently from how it came out
        last = bool(self.checkpoints.slots) and final
        # one token runs the decode program (prefill's would compute a whole padded chunk); with an MTP block it
        # leaves the MTP's KV for that row unwritten, which only makes the next first draft less likely to hold
        if len(ids) == 1:
            if last:
                self._checkpoint(prompt=True)
            logits = self.session.step(ids)
        else:
            # where the session pads (tinyllm.tg.upstream), the stretch that ends the prompt is one call, and the
            # checkpoint is the state that call keeps 256 rows before its end
            hook = functools.partial(self._checkpoint, True) if last else None
            logits = self.session.prefill(
                ids, wait_each=False, following=following, before_last=hook, ends=final, inside_last=hook
            )
            self._written_end = max(self._written_end, self.session.padded_end)
        self._extend(ids)
        if len(self.session.trace) > 2 * TRACE_KEEP:
            del self.session.trace[:-TRACE_KEEP]
        return np.asarray(logits[0])

    def _advance(self, target: int) -> np.ndarray:
        """Bring the state to the prompt's first `target` tokens, by `begin`'s plan."""
        prompt, n = self._prompt, self._prompt_end
        assert prompt is not None
        logits = None
        while len(self.tokens) < target:
            here = len(self.tokens)
            if self._jumps and self._jumps[0][0] == here:  # the caches from here to the checkpoint are valid
                self._load(self._jumps.pop(0)[1])
                continue
            stop = min(target, self._jumps[0][0]) if self._jumps else target
            logits = self._run_prompt(prompt[here:stop], prompt[stop] if stop < n else None, stop == n)
        # nothing run: a jump had already taken the state past `target`
        return logits if logits is not None else np.empty(0, dtype=np.float32)

    def feed(self, tokens: Sequence[int], following: int | None = None) -> np.ndarray:
        """The logits after the last token; the array is overwritten by the next call. `following`: the token that
        comes next when the caller already knows it (the rest of a prompt), for the MTP block's KV.

        The prompt given to `begin` is handed over in order, in any slices, from the position `begin` returned.
        A slice that a jump of the plan already covers runs nothing and returns an empty array."""
        ids = [int(t) for t in tokens]
        if not ids:
            raise ValueError("feed needs at least one token")
        prompt, fed = self._prompt, self._fed
        if prompt is not None and fed < self._prompt_end and prompt[fed : fed + len(ids)] == ids:
            self._fed = fed + len(ids)
            return self._advance(self._fed)
        # generated tokens, or tokens `begin` was not told about: they follow what the state holds
        self._jumps, self._fed = [], self._prompt_end
        if len(ids) == 1:
            logits = self.session.step(ids)
        else:
            logits = self.session.prefill(ids, wait_each=False, following=following)
        self._extend(ids)
        self._after_decode()
        return np.asarray(logits[0])

    def feed_sampled(self, token: int, params: Sequence[float]) -> np.ndarray:
        """Run one token and sample the next on the GPU: the packed result of `tinyllm.tg.sampler.sample_row`.

        `params`: temperature, top_k, top_p, min_p, u (a uniform number in [0, 1)). The array is reused by the
        next call."""
        out = self.session.decode_sampled([int(token)], params)
        self._extend([int(token)])
        self._after_decode()
        return np.asarray(out)

    @property
    def last_drafted(self) -> int:
        """Drafts of the last cycle: K, or those of a block copied from earlier text."""
        return int(self.session.last_drafted)

    def feed_speculative(
        self, token: int, params: Sequence[Sequence[float]], copy: Sequence[int] | None = None
    ) -> tuple[np.ndarray, int]:
        """One draft-and-verify cycle after `token` (tinyllm.tg.speculative): device-sampler results per verify
        row and n; rows 0..n are the n + 1 tokens produced. `params`: `speculative_rows` rows of settings. `copy`
        forces a copied block of those tokens (warming up). The array is reused by the next call."""
        first = len(self.tokens)
        rows, n = self.session.speculate(int(token), params, copy)
        self._written_end = max(self._written_end, first + self.session.last_drafted + 1)  # the verify's rows
        self._extend([int(token), *(int(t) for t in rows[:n, 0])])
        self._after_decode()
        return rows, n

    def warmup(self) -> None:
        """Compile the prefill programs (the full chunk's, those of the shorter fixed lengths and those for partial
        chunks), both decode programs (logits out, and sampled on the GPU) and, with speculative decoding, the
        cycle's and the restore programs (each runs at least twice, the second run captures it), then forget the
        state. Programs loaded from disk (`session.programs`) run once; those built here are saved there."""
        start = time.perf_counter()
        programs = self.session.programs
        spent: dict[str, float] = {}  # seconds by stage, over both passes: which programs a start pays for

        def stage(name: str, run: Any, *args: Any) -> None:
            t = time.perf_counter()
            run(*args)
            synchronize()
            spent[name] = spent.get(name, 0.0) + time.perf_counter() - t

        sampler = [1.0, 20, 0.95, 0.0, 0.5]
        rows, block = [sampler] * self.speculative_rows, [0] * self.copy

        def cycles(*copy: list[int]) -> None:
            for _ in range(3):
                self.feed_speculative(0, rows, *copy)

        for again in range(2):
            if again and not programs.pending():  # every program was loaded: none waits for its capturing call
                break
            self.begin([0])
            stage(f"prefill {self.chunk} and the smallest partial", self.feed, [0] * (self.chunk + 3))
            sizes = self.session.partial_sizes  # one token more than each takes runs the next program up
            for size, above in zip(sizes, [*sizes[1:], self.chunk], strict=False):  # none: nothing to run
                if len(self.tokens) + size + 8 + self.speculative_rows < self.max_context:
                    stage(f"prefill up to {above}", self.feed, [0] * (size + 1))
            for size in self.session.static_sizes:  # the fixed-length programs of calls on the checkpoint grid
                if len(self.tokens) + size + 8 + self.speculative_rows < self.max_context:
                    stage(f"prefill {size}", self.feed, [0] * size)
            stage("decode", lambda: (self.feed([0]), self.feed([0])))
            if self.device_sampling:
                stage("decode", lambda: (self.feed_sampled(0, sampler), self.feed_sampled(0, sampler)))
            if self.speculative:  # draft-and-verify cycles, and the restore program for each rejected row
                stage("draft and verify", cycles)
                if self.copy:  # the verify of a copied block
                    stage("copied block", cycles, block)
                for n in range(self.speculative_rows - 1):
                    self.session._restore_row = n
                    stage("restores", self.session._settle)
        self.warmup_stages = spent
        if self.checkpoints.slots:  # the second call captures each program, the third replays it with another slot
            other = min(1, self.checkpoints.slots - 1)
            for slot in (0, other, 0):
                self.session.save_state(slot)
            for slot in (0, other, 0):
                self.session.load_state(slot, self.session.pos)
            if self.session.pads:  # the copy of the state a padded prefill call keeps
                for slot in (0, other, 0):
                    self.session.save_mid_state(slot)
        self.forget()
        self.session.trace.clear()
        programs.save()
        self.warmup_seconds = time.perf_counter() - start
