"""Prompt reuse past a divergence: the checkpoint bookkeeping (tinyllm.runner.checkpoints) on its own, and the
runner resuming from a checkpoint on the development model (CPU), which must give exactly the logits that running
the same calls without the detour gives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tinyllm.runner.checkpoints import Checkpoints, common_prefix


def test_common_prefix() -> None:
    assert common_prefix([1, 2, 3], [1, 2, 3, 4]) == 3
    assert common_prefix([1, 2, 3], [1, 5, 3]) == 1
    assert common_prefix([], [1]) == 0
    assert common_prefix([7, 8], [9]) == 0


def test_best_is_the_newest_at_or_below_the_limit() -> None:
    c = Checkpoints(4)
    for pos in (0, 96, 200, 456):
        c.add(pos, prompt=pos != 456)
    assert [x.pos for x in c.items] == [96, 200, 456]  # position 0 is a reset, never a checkpoint
    assert c.best(455).pos == 200  # type: ignore[union-attr]
    assert c.best(456).pos == 456  # type: ignore[union-attr]
    assert c.best(95) is None
    assert c.add(200, prompt=True) is None  # already there
    with pytest.raises(ValueError):
        c.add(100, prompt=True)  # behind the newest: the sequence only grows between truncations


def test_truncate_drops_what_a_restore_makes_stale() -> None:
    c = Checkpoints(4)
    for pos in (96, 200, 456):
        c.add(pos, prompt=True)
    c.truncate(200)
    assert [x.pos for x in c.items] == [96, 200]
    slot = c.add(300, prompt=False)
    assert slot is not None and slot not in {x.slot for x in c.items[:-1]}


def test_eviction_keeps_the_newest_prompt_checkpoint() -> None:
    """A long reply's decode checkpoints push out older ones, but not the one where the reply starts, which the
    next request resumes from when the template re-renders the reply."""
    c = Checkpoints(3)
    c.add(100, prompt=True)
    c.add(1000, prompt=True)  # this request's prompt
    for pos in (1256, 1512, 1768, 2024):
        assert c.add(pos, prompt=False) is not None
    assert [(x.pos, x.prompt) for x in c.items] == [(1000, True), (1768, False), (2024, False)]
    assert len({x.slot for x in c.items}) == 3
    c.add(2300, prompt=True)  # the next prompt: the previous one is no longer protected
    assert [x.pos for x in c.items] == [1768, 2024, 2300]


def test_one_slot_holds_the_prompt_checkpoint() -> None:
    c = Checkpoints(1)
    assert c.add(100, prompt=True) == 0
    assert c.add(356, prompt=False) is None
    assert [x.pos for x in c.items] == [100]
    assert Checkpoints(0).add(100, prompt=True) is None


@pytest.fixture(scope="module")
def runner(dev_model: Path) -> Any:
    from tinyllm.runner.upstream import UpstreamRunner

    r = UpstreamRunner(dev_model, max_context=1024, chunk=32, checkpoints=4, checkpoint_every=16)
    r.warmup()
    return r


def _fresh(runner: Any, prompt_end: int) -> None:
    """Forget the state and every checkpoint, as for a new conversation of `prompt_end` prompt tokens."""
    runner.begin([0])  # shares nothing with what is held: starts over
    runner._prompt_end = prompt_end


def _run(runner: Any, prompt: list[int], decode: int) -> tuple[int, list[int], np.ndarray]:
    """What the engine does with a greedy request: begin, feed the rest of the prompt, decode. Returns how many
    tokens were reused, the generated tokens and the logits after the prompt."""
    cached = runner.begin(prompt)
    logits = runner.feed(prompt[cached:]).copy()
    first, out = logits, []
    for _ in range(decode):
        out.append(int(logits.argmax()))
        logits = runner.feed([out[-1]])
    return cached, out, first


@pytest.mark.model
def test_resuming_from_a_checkpoint_matches_the_calls_it_replaces(runner: Any) -> None:
    tok = runner.session.tokenizer()
    a = tok.encode("Write a Rust function that reverses a linked list in place, with a short explanation. " * 4)
    assert 64 < len(a) < 96, len(a)
    # request 1: the prompt in chunks of 32 (a checkpoint before the last one, at 64), then 40 tokens decoded
    # (checkpoints every 16 tokens past the newest)
    _, reply, _ = _run(runner, a, 40)
    held = list(runner.tokens)
    positions = [c.pos for c in runner.checkpoints.items]
    assert 64 in positions and any(p > len(a) for p in positions), positions

    # request 2 departs where the reply starts: it resumes at 64, below the departure, and feeds the rest
    other = tok.encode(" Use an iterator instead.")
    b = a + other
    cached, _, first_b = _run(runner, b, 8)
    assert cached == 64
    # the same calls from scratch: the prompt in the same chunks (0-32-64, then the rest from 64)
    _fresh(runner, len(b))
    runner.feed(b[:64], following=b[64])
    ref = runner.feed(b[64:]).copy()
    assert np.array_equal(first_b, ref)

    # request 3 keeps part of request 1's reply: it resumes at a decode checkpoint inside it
    _run(runner, a, 40)
    assert runner.tokens == held
    stop = len(a) + 30
    c = runner.begin(held[:stop] + other)
    decode_points = [p for p in positions if len(a) < p <= stop]
    assert c == max(decode_points), (c, positions)
    resumed = runner.feed((held[:stop] + other)[c:]).copy()
    # the same calls without the detour: request 1 up to that checkpoint, then the rest
    _fresh(runner, len(a))
    logits = runner.feed(a)
    for t in reply[: c - len(a)]:
        assert int(logits.argmax()) == t
        logits = runner.feed([t])
    runner._prompt_end = len(held[:stop] + other)
    assert np.array_equal(resumed, runner.feed((held[:stop] + other)[c:]))


@pytest.mark.model
def test_a_shorter_or_unrelated_prompt(runner: Any) -> None:
    tok = runner.session.tokenizer()
    a = tok.encode("Explain borrowing in Rust with one example. " * 8)
    assert len(a) > 32, len(a)
    _run(runner, a, 20)
    # the same prompt again (a retry): resumes at the checkpoint before its last chunk, not at 0
    cached, _, _ = _run(runner, a, 1)
    assert cached == (len(a) - 1) // 32 * 32
    # nothing in common: starts over
    cached, _, _ = _run(runner, tok.encode("Bonjour"), 1)
    assert cached == 0 and not runner.checkpoints.items
