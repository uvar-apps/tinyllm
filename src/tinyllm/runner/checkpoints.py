"""Which positions of the held sequence have a saved recurrent state (prompt reuse past a divergence).

Attention layers index their cache by position, so going back to an earlier position costs them nothing: entries
past it are masked out and overwritten. The recurrent (DeltaNet) layers fold every token into one state, which
can only be put back from a copy taken at that exact position. The runner takes such copies (checkpoints) at a few
positions and, when a new prompt shares only part of what the state holds, resumes from the newest one at or below
the shared part.

The case this is for: a coding agent sends the conversation again with the model's last reply re-rendered by the
chat template, usually without its reasoning, so the new prompt departs from the held tokens where that reply
starts. A checkpoint taken just before each prompt's last prefill call lies at most one chunk before that point.
Checkpoints taken every few hundred tokens while decoding cover a reply that is re-rendered with its reasoning but
differs later on (in the formatting of a tool call, say).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CHECKPOINTS = 8  # slots kept by `tinyllm serve`: 157 MB each on Qwen3.8-27B (48 DeltaNet blocks)
CHECKPOINT_EVERY = 256  # generated tokens between checkpoints while decoding


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """How many leading tokens `a` and `b` share."""
    n = min(len(a), len(b))
    if list(a[:n]) == list(b[:n]):
        return n
    return int(np.flatnonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0])


@dataclass(frozen=True)
class Checkpoint:
    pos: int  # the state holds the first `pos` tokens
    slot: int  # the buffer that holds the copy
    prompt: bool  # taken before a prompt's last prefill call (else while decoding)


class Checkpoints:
    """The checkpoints of the held sequence: each holds the state after a prefix of it, in one of `slots` buffers.

    Positions only grow while the sequence is extended; `truncate` drops the ones a restore makes stale. When every
    slot is taken, the lowest position goes first, except the newest prompt checkpoint, which the next request of an
    agent loop is most likely to need.
    """

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.items: list[Checkpoint] = []  # ascending positions

    def clear(self) -> None:
        self.items = []

    def truncate(self, pos: int) -> None:
        """Forget the checkpoints past `pos`: the state no longer continues the way they recorded."""
        self.items = [c for c in self.items if c.pos <= pos]

    def best(self, limit: int) -> Checkpoint | None:
        """The checkpoint at the highest position at most `limit`, above 0 (position 0 is a reset)."""
        found = [c for c in self.items if 0 < c.pos <= limit]
        return found[-1] if found else None

    @property
    def newest(self) -> int:
        """The position of the newest checkpoint, 0 when there is none."""
        return self.items[-1].pos if self.items else 0

    def add(self, pos: int, prompt: bool) -> int | None:
        """Record a checkpoint at `pos` and return the slot to copy the state into; None when there is nothing to
        do (no slots, position 0, or a checkpoint at `pos` already)."""
        if not self.slots or pos <= 0 or any(c.pos == pos for c in self.items):
            return None
        if any(c.pos > pos for c in self.items):
            raise ValueError(f"checkpoint at {pos} behind the newest one at {self.newest}")
        used = {c.slot for c in self.items}
        free = [s for s in range(self.slots) if s not in used]
        if free:
            slot = free[0]
        else:
            prompts = [c for c in self.items if c.prompt]
            keep = prompts[-1] if prompts and not prompt else None
            victims = [c for c in self.items if c is not keep]
            if not victims:  # one slot, holding the prompt checkpoint
                return None
            self.items.remove(victims[0])
            slot = victims[0].slot
        self.items.append(Checkpoint(pos, slot, prompt))
        return slot
