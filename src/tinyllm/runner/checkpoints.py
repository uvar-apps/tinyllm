"""Prompt reuse: which saved recurrent states and which cache positions a new prompt can use.

Attention layers index their cache by position, so going back to an earlier position costs them nothing: entries
past it are masked out and overwritten. The recurrent (DeltaNet) layers fold every token into one state, which
can only be put back from a copy taken at that exact position. The runner takes such copies (checkpoints) at a few
positions and, when a new prompt shares only part of what the state holds, resumes from the newest one at or below
the shared part.

Two cases this is for:

- A coding agent sends the conversation again with the model's last reply re-rendered by the chat template,
  usually without its reasoning, so the new prompt departs from the held tokens where that reply starts. A
  checkpoint taken just before each prompt's last prefill call lies at most one chunk before that point.
  Checkpoints taken every few hundred tokens while decoding cover a reply that is re-rendered with its reasoning
  but differs later on (in the formatting of a tool call, say).
- A client sends other requests between the turns of a conversation (Jan asks for a chat title after every reply;
  agents summarise or classify on the side). Such a request overwrites the first positions of the caches and
  replaces the recurrent state, but the conversation's checkpoints and its cache entries past the request's length
  are still good. So a checkpoint remembers the tokens it stands for and matches any prompt that starts with them
  (`Checkpoints`), the runner records which token sequence each cache position was computed from (`Ownership`),
  and `plan` finds the cheapest way back: run the positions the other request overwrote again, then jump to the
  conversation's deepest checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CHECKPOINTS = 8  # slots kept by `tinyllm serve`: 157 MB each on Qwen3.8-27B (48 DeltaNet blocks)
CHECKPOINT_EVERY = 256  # generated tokens between checkpoints while decoding


def common_prefix(a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray) -> int:
    """How many leading tokens `a` and `b` share."""
    n = min(len(a), len(b))
    x, y = np.asarray(a[:n]), np.asarray(b[:n])
    differ = np.flatnonzero(x != y)
    return int(differ[0]) if len(differ) else n


@dataclass
class Checkpoint:
    pos: int  # the state holds the first `pos` tokens
    slot: int  # the buffer that holds the copy
    prompt: bool  # taken before a prompt's last prefill call (else while decoding)
    tokens: np.ndarray  # those `pos` tokens
    epoch: int  # the request that took it or last matched it


class Checkpoints:
    """Saved recurrent states, each in one of `slots` buffers, each for the token prefix it remembers.

    When every slot is taken, the checkpoint that goes is the one of the oldest request (a request refreshes the
    checkpoints its prompt matches); among those of one request a decoding checkpoint goes before a prompt one, and
    a lower position before a higher. A new checkpoint that would itself be the first to go is not taken.
    """

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.items: list[Checkpoint] = []

    def clear(self) -> None:
        self.items = []

    def matching(self, tokens: Sequence[int] | np.ndarray, limit: int) -> list[Checkpoint]:
        """The checkpoints of prefixes of `tokens`, at positions 1 .. `limit`, ascending."""
        ids = np.asarray(tokens)
        found = [c for c in self.items if 0 < c.pos <= min(limit, len(ids)) and np.array_equal(c.tokens, ids[: c.pos])]
        return sorted(found, key=lambda c: c.pos)

    def touch(self, found: Sequence[Checkpoint], epoch: int) -> None:
        """Mark `found` as used by request `epoch`."""
        for c in found:
            c.epoch = epoch

    @staticmethod
    def _rank(c: Checkpoint) -> tuple[int, bool, int]:
        return c.epoch, c.prompt, c.pos

    def add(self, pos: int, prompt: bool, tokens: Sequence[int] | np.ndarray, epoch: int) -> int | None:
        """Record a checkpoint of `tokens[:pos]` and return the slot to copy the state into; None when there is
        nothing to do (no slots, position 0, that checkpoint already taken, or nothing less useful to replace)."""
        if not self.slots or pos <= 0:
            return None
        prefix = np.array(tokens[:pos], dtype=np.int64)
        if len(prefix) != pos:
            raise ValueError(f"a checkpoint at {pos} of {len(prefix)} tokens")
        for c in self.items:
            if c.pos == pos and np.array_equal(c.tokens, prefix):
                c.epoch, c.prompt = epoch, c.prompt or prompt
                return None
        new = Checkpoint(pos, -1, prompt, prefix, epoch)
        free = sorted(set(range(self.slots)) - {c.slot for c in self.items})
        if free:
            new.slot = free[0]
        else:
            victim = min(self.items, key=self._rank)
            if self._rank(victim) >= self._rank(new):
                return None
            self.items.remove(victim)
            new.slot = victim.slot
        self.items.append(new)
        return new.slot


class Ownership:
    """Which token sequence the caches of each position were computed from.

    A position's cache entries (attention keys and values, the sparse attention's indexer keys) depend on every
    token up to it, so they serve a prompt only if the sequence that wrote them agrees with the prompt up to that
    position. Segments (first, end, tokens) say: positions first .. end - 1 were written while the state held
    `tokens` (at least `end` of them).
    """

    def __init__(self) -> None:
        self.segments: list[tuple[int, int, np.ndarray]] = []

    def clear(self) -> None:
        self.segments = []

    def wrote(self, first: int, end: int, tokens: np.ndarray, spill: int = 0) -> None:
        """Positions first .. end - 1 now hold `tokens`' entries; up to `spill` positions after them may have been
        overwritten too (rows of a speculative verify that were not accepted) and belong to no one."""
        if end <= first:
            return
        lost, kept = end + spill, []
        for a, b, owner in self.segments:
            if a < first:
                kept.append((a, min(b, first), owner))
            if b > lost:
                kept.append((max(a, lost), b, owner))
        kept.append((first, end, tokens))
        kept.sort(key=lambda s: s[0])
        merged: list[tuple[int, int, np.ndarray]] = []
        for a, b, owner in kept:
            if a >= b:
                continue
            if merged and merged[-1][1] == a and common_prefix(merged[-1][2], owner) >= a:
                merged[-1] = (merged[-1][0], b, owner)  # the later sequence continues the earlier one's prefix
            else:
                merged.append((a, b, owner))
        self.segments = merged

    def valid(self, tokens: Sequence[int] | np.ndarray) -> np.ndarray:
        """For each position of `tokens`: whether the caches hold that prompt's entries there."""
        ids = np.asarray(tokens)
        ok = np.zeros(len(ids), dtype=bool)
        for a, b, owner in self.segments:
            end = min(b, common_prefix(owner, ids))
            if end > a:
                ok[a:end] = True
        return ok


def plan(valid: np.ndarray, checkpoints: Sequence[int]) -> tuple[int, list[tuple[int, int]], int]:
    """How to bring the state to a prompt with the fewest tokens run.

    `valid[p]`: the caches hold the prompt's entries at position p. `checkpoints`: ascending positions (1 .. the
    prompt's length - 1) with a saved state of the prompt's first tokens. Returns (start, jumps, run):

      start  the checkpoint to load first, 0 to start over;
      jumps  [(at, pos)]: once the state holds `at` tokens, load the checkpoint at `pos` (past `at`: the caches in
             between are valid) and go on from there;
      run    how many tokens are run in all. The last token always is: its logits are wanted.
    """
    n = len(valid)
    if n == 0:
        raise ValueError("a prompt has at least one token")
    bad = [*np.flatnonzero(~valid[: n - 1]).tolist(), n - 1]
    cur, start, started, jumps, run, i = 0, 0, False, [], 0, 0
    while cur < n:
        while bad[i] < cur:
            i += 1
        below = [c for c in checkpoints if cur < c <= bad[i]]  # every position from `cur` to it is valid
        if below:
            if started:
                jumps.append((cur, below[-1]))
            else:
                start = below[-1]
            cur, started = below[-1], True
            continue
        started = True
        while i + 1 < len(bad) and bad[i + 1] == bad[i] + 1:  # through the run of positions to compute
            i += 1
        run += bad[i] + 1 - cur
        cur = bad[i] + 1
    return start, jumps, run
