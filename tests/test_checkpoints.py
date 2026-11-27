"""Prompt reuse past a divergence: the checkpoint bookkeeping (tinyllm.runner.checkpoints) on its own, and the
runner resuming from a checkpoint on the development model (CPU), which must give exactly the logits that running
the same calls without the detour gives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tinyllm import tg  # noqa: F401  (chooses the device before a test imports tinygrad)
from tinyllm.runner.checkpoints import Checkpoints, Ownership, common_prefix, plan


def test_common_prefix() -> None:
    assert common_prefix([1, 2, 3], [1, 2, 3, 4]) == 3
    assert common_prefix([1, 2, 3], [1, 5, 3]) == 1
    assert common_prefix([], [1]) == 0
    assert common_prefix([7, 8], [9]) == 0


def test_checkpoints_match_the_prompts_that_start_with_their_tokens() -> None:
    c = Checkpoints(4)
    a, b = list(range(100, 600)), list(range(700, 1200))
    assert c.add(0, True, a, 1) is None  # position 0 is a reset, never a checkpoint
    for pos in (96, 200):
        assert c.add(pos, True, a, 1) is not None
    assert c.add(96, True, b, 2) is not None  # another sequence at the same position
    assert c.add(200, False, a, 3) is None  # already there: refreshed
    assert [x.pos for x in c.matching(a, 455)] == [96, 200]
    assert [x.pos for x in c.matching(a, 199)] == [96]
    assert [x.pos for x in c.matching(a[:150] + [1] * 100, 249)] == [96]  # departs at 150
    assert [x.slot for x in c.matching(b, 400)] != [x.slot for x in c.matching(a, 100)]
    assert c.matching([1, 2, 3], 2) == []
    assert {x.pos: x.epoch for x in c.matching(a, 455)} == {96: 1, 200: 3}
    with pytest.raises(ValueError):
        c.add(900, True, a, 4)  # more tokens than the sequence has


def test_eviction_takes_the_oldest_request_then_decode_checkpoints() -> None:
    """A long reply's decode checkpoints push out older ones, but not the one where the reply starts, which the
    next request resumes from when the template re-renders the reply."""
    c, seq = Checkpoints(3), list(range(4000))
    c.add(100, True, seq, 1)
    c.add(1000, True, seq, 2)  # this request's prompt
    for pos in (1256, 1512, 1768, 2024):
        assert c.add(pos, False, seq, 2) is not None
    assert sorted((x.pos, x.prompt) for x in c.items) == [(1000, True), (1768, False), (2024, False)]
    assert len({x.slot for x in c.items}) == 3
    c.add(2300, True, seq, 3)  # the next prompt: a decode checkpoint of the request before goes first
    assert sorted(x.pos for x in c.items) == [1000, 2024, 2300]


def test_a_side_request_does_not_push_out_the_conversation_s_prompt_checkpoint() -> None:
    c, conv, side = Checkpoints(3), list(range(9000)), list(range(20000, 20500))
    c.add(7680, True, conv, 1)
    c.add(7936, False, conv, 1)
    c.add(8192, False, conv, 1)
    assert c.add(256, True, side, 2) is not None  # takes a decode checkpoint's slot
    assert sorted(x.pos for x in c.items) == [256, 7680, 8192]
    found = c.matching(conv[:8000], 7999)
    assert [x.pos for x in found] == [7680]
    c.touch(found, 3)
    assert c.add(8448, True, conv, 3) is not None  # the conversation's next turn: its old decode checkpoint goes
    assert sorted(x.pos for x in c.items) == [256, 7680, 8448]
    assert c.add(8704, False, conv, 3) is not None  # then the side request's
    assert sorted(x.pos for x in c.items) == [7680, 8448, 8704]


def test_one_slot_holds_the_prompt_checkpoint() -> None:
    c, seq = Checkpoints(1), list(range(400))
    assert c.add(100, True, seq, 1) == 0
    assert c.add(356, False, seq, 1) is None
    assert [x.pos for x in c.items] == [100]
    assert Checkpoints(0).add(100, True, seq, 1) is None


def test_ownership_tracks_which_sequence_wrote_each_position() -> None:
    conv = np.arange(1000, 1800)
    side = np.arange(5000, 5100)
    own = Ownership()
    own.wrote(0, 700, conv)
    assert own.valid(conv[:800]).tolist() == [True] * 700 + [False] * 100
    own.wrote(0, 100, side)  # another request over the first positions
    ok = own.valid(conv[:800])
    assert not ok[:100].any() and ok[100:700].all() and not ok[700:].any()
    assert own.valid(side).all()
    # a prompt that departs from the conversation at 300 can use its positions up to there
    other = np.concatenate([conv[:300], np.zeros(50, dtype=np.int64)])
    ok = own.valid(other)
    assert not ok[:100].any() and ok[100:300].all() and not ok[300:].any()
    # the conversation runs its first positions again, then goes on: one segment again
    own.wrote(0, 100, conv[:100])
    own.wrote(700, 800, conv, spill=4)
    assert own.valid(conv[:800]).all() and len(own.segments) == 1
    # rows a speculative verify wrote past the end belong to no one
    longer = np.arange(1000, 1900)
    own.wrote(0, 850, longer)
    own.wrote(300, 310, np.concatenate([longer[:300], np.full(10, 7)]), spill=3)
    ok = own.valid(longer[:850])
    assert ok[:300].all() and not ok[300:313].any() and ok[313:].all()


def test_plan_runs_only_what_the_caches_and_checkpoints_do_not_cover() -> None:
    def valid(n: int, *ranges: tuple[int, int]) -> np.ndarray:
        ok = np.zeros(n, dtype=bool)
        for a, b in ranges:
            ok[a:b] = True
        return ok

    # nothing to reuse
    assert plan(valid(100), []) == (0, [], 100)
    # the reply was re-rendered: resume at the deepest checkpoint below the departure
    assert plan(valid(1000, (0, 800)), [256, 768]) == (768, [], 232)
    # a checkpoint past the departure is not one of this prompt's: the caller only passes matching ones
    # another request overwrote the first 583 positions: run them again, then jump
    assert plan(valid(9000, (583, 7801)), [7680]) == (0, [(583, 7680)], 583 + 1320)
    # ... and with a checkpoint below the overwritten stretch, start there
    assert plan(valid(9000, (0, 100), (583, 7801)), [64, 7680]) == (64, [(583, 7680)], 519 + 1320)
    # no checkpoint to jump to: the valid stretch is run again too
    assert plan(valid(1000, (100, 900)), []) == (0, [], 1000)
    # everything valid and a checkpoint at the last position: only the last token is run
    assert plan(valid(500, (0, 500)), [256, 499]) == (499, [], 1)
    assert plan(valid(500, (0, 500)), [256]) == (256, [], 244)
    # two overwritten stretches with a checkpoint between them
    assert plan(valid(3000, (100, 1000), (1200, 2500)), [512, 2048]) == (0, [(100, 512), (1200, 2048)], 100 + 688 + 952)
    with pytest.raises(ValueError):
        plan(valid(0), [])


def test_a_prompt_s_last_stretch_is_one_padded_call_where_programs_take_padding() -> None:
    from tinyllm.tg.upstream import prefill_call

    def calls(pos: int, end: int, padded: tuple[int, ...], limit: int = 32768) -> list[tuple[int, int]]:
        tail, out = (end - 1) // 256 * 256, []  # the checkpoint before the last call, where calls are cut
        while pos < end:
            n, length = prefill_call(pos, end, 2048, tail, padded, limit)
            out.append((n, length))
            pos += n
        return out

    lengths = (512, 768, 1024, 1280, 1536, 1792, 2048)
    # an agent's step of 1,587 new tokens after a checkpoint: a call to the next checkpoint and a short one, or one
    assert calls(7168, 8755, ()) == [(1536, 0), (51, 0)]
    assert calls(7168, 8755, lengths) == [(1587, 1792)]
    # whole chunks first; a stretch of exactly a program's length runs unpadded through the same program
    assert calls(0, 3000, lengths) == [(2048, 0), (952, 1024)]
    assert calls(0, 2048, lengths) == [(2048, 2048)]
    assert calls(0, 512, lengths) == [(512, 512)]
    # a last call of at most 256 tokens runs as it is, with the checkpoint before it
    assert calls(0, 2200, lengths) == [(2048, 0), (152, 0)]
    assert calls(0, 256, lengths) == [(256, 0)]
    # no program of the padded length, or no room for its rows below the context's end: cut as without padding
    assert calls(2048, 3000, (768, 2048)) == [(768, 0), (184, 0)]
    assert calls(30720, 32700, lengths) == [(1980, 2048)]
    assert calls(31000, 32700, lengths) == [(1512, 0), (188, 0)]
    # from a position off the grid (a checkpoint taken while decoding): a whole chunk instead of a cut at the grid
    assert calls(100, 2300, lengths) == [(2048, 0), (152, 0)]
    assert calls(100, 2300, ()) == [(1948, 0), (252, 0)]


def test_checkpoints_hold_the_n_gram_convolution_state() -> None:
    """Qwen3.8-Flash-Next: the state a checkpoint copies is the DeltaNet blocks' and the n-gram embedding's
    convolution state; the attention and indexer caches are indexed by position. Resuming cuts the n-gram history
    back to the checkpoint. Checked on stand-in blocks: no model on the CPU has an n-gram embedding."""
    from types import SimpleNamespace

    from tinygrad import Tensor
    from tinygrad.llm.model import GatedDeltaNetBlock, TransformerBlock

    from tinyllm.tg.upstream import UpstreamSession

    ple, other = SimpleNamespace(), SimpleNamespace()  # stand-ins for tinygrad's PLELayer
    deltanet = [object.__new__(GatedDeltaNetBlock) for _ in range(3)]
    for b in deltanet:
        b.recurrent_state, b.conv_state = Tensor.zeros(2, 2), Tensor.zeros(3, 2)
    deltanet[1].ple = ple
    s = object.__new__(UpstreamSession)
    s.model = SimpleNamespace(config=SimpleNamespace(ssm=SimpleNamespace(kda=False)), blk=[*deltanet])
    s.model.blk.append(object.__new__(TransformerBlock))
    s._ple = ple
    assert s._checkpointable()
    with pytest.raises(RuntimeError):  # before the first call there is no n-gram state yet
        s._state_tensors()
    ple.conv_state = Tensor.zeros(9, 4)
    states = s._state_tensors()
    assert len(states) == 7 and states[-1] is ple.conv_state
    deltanet[1].ple = other  # n-gram state the session does not know about
    assert not s._checkpointable()

    s.max_context, s.speculative, s._slots, s._history = 4096, 0, [[]], list(range(300))
    s._load_jit = lambda *slot: None
    s.load_state(0, 200)
    assert s.pos == 200 and s._history == list(range(200))
    s.load_state(0, 120, history=list(range(500, 700)))  # a checkpoint of another sequence than the one held
    assert s.pos == 120 and s._history == list(range(500, 620))


@pytest.fixture(scope="module")
def runner(dev_model: Path) -> Any:
    from tinyllm.runner.upstream import UpstreamRunner

    r = UpstreamRunner(dev_model, max_context=1024, chunk=32, checkpoints=4, checkpoint_every=16)
    r.warmup()
    return r


def _run(runner: Any, prompt: list[int], decode: int) -> tuple[int, list[int], np.ndarray]:
    """What the engine does with a greedy request: begin, feed the rest of the prompt, decode. Returns where the
    prompt was fed from, the generated tokens and the logits after the prompt."""
    cached = runner.begin(prompt)
    logits = runner.feed(prompt[cached:]).copy()
    first, out = logits, []
    for _ in range(decode):
        out.append(int(logits.argmax()))
        logits = runner.feed([out[-1]])
    return cached, out, first


@pytest.mark.model
def test_resuming_from_a_checkpoint_matches_the_calls_it_replaces(runner: Any) -> None:
    runner.forget()
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
    assert cached == 64 and runner.reused == 64
    # the same calls from scratch: the prompt in the same chunks (0-32-64, then the rest from 64)
    runner.forget()
    assert runner.begin(b) == 0
    runner.feed(b[:64], following=b[64])
    ref = runner.feed(b[64:]).copy()
    assert np.array_equal(first_b, ref)

    # request 3 keeps part of request 1's reply: it resumes at a decode checkpoint inside it
    runner.forget()
    _run(runner, a, 40)
    assert runner.tokens == held
    stop = len(a) + 30
    third = held[:stop] + other
    c = runner.begin(third)
    decode_points = [p for p in positions if len(a) < p <= stop]
    assert c == max(decode_points), (c, positions)
    resumed = runner.feed(third[c:]).copy()
    # the same calls without the detour: request 1 up to that checkpoint, then the rest
    runner.forget()
    runner.begin(a)
    logits = runner.feed(a)
    for t in reply[: c - len(a)]:
        assert int(logits.argmax()) == t
        logits = runner.feed([t])
    assert runner.begin(third) == c  # extends what the state holds: no checkpoint is loaded
    assert np.array_equal(resumed, runner.feed(third[c:]))


@pytest.mark.model
def test_a_shorter_or_unrelated_prompt(runner: Any) -> None:
    runner.forget()
    tok = runner.session.tokenizer()
    a = tok.encode("Explain borrowing in Rust with one example. " * 8)
    assert len(a) > 32, len(a)
    _run(runner, a, 20)
    # the same prompt again (a retry): resumes at the checkpoint before its last chunk, not at 0
    cached, _, _ = _run(runner, a, 1)
    assert cached == (len(a) - 1) // 32 * 32
    # nothing in common: starts over, and the conversation's checkpoints stay
    cached, _, _ = _run(runner, tok.encode("Bonjour"), 1)
    assert cached == 0 and runner.reused == 0
    assert any(c.pos == (len(a) - 1) // 32 * 32 for c in runner.checkpoints.items)


@pytest.mark.model
@pytest.mark.parametrize("slices", [(), (40,), (32, 48)], ids=["one feed", "two feeds", "a feed the jump covers"])
def test_another_request_between_two_turns_of_a_conversation(runner: Any, slices: tuple[int, ...]) -> None:
    """A client asks for something else between the turns (a chat title, say): the conversation resumes from its
    own checkpoint once the positions the other request overwrote are run again, with the logits the conversation
    gets without the interruption."""
    tok = runner.session.tokenizer()
    a = tok.encode("Describe how a hash map resolves collisions, with a small example in Rust. " * 7)
    side = tok.encode("Summarize the conversation in five words. " * 6)[:32]
    other = tok.encode(" Now compare open addressing with chaining.")
    assert 100 < len(a) < 160 and len(side) == 32 and side[0] != a[0], (len(a), len(side))
    b = a + other
    last = (len(a) - 1) // 32 * 32  # the checkpoint before turn 1's last prefill call

    # without the interruption: turn 1, then turn 2 resumes at turn 1's checkpoint
    runner.forget()
    _run(runner, a, 12)
    cached, _, ref = _run(runner, b, 0)
    assert cached == last

    # with a request of one chunk in between: positions 0..31 are run again (the same call as in turn 1), then the
    # state jumps to the checkpoint
    runner.forget()
    _run(runner, a, 12)
    assert _run(runner, side, 0)[0] == 0
    start = runner.begin(b)
    assert start == 0 and runner.reused == last - 32, (start, runner.reused)
    logits, fed = None, start
    for end in (*slices, len(b)):
        got = runner.feed(b[fed:end])
        logits, fed = got.copy() if len(got) else logits, end
    assert runner.tokens == b
    assert np.array_equal(logits, ref)
    # the side request again, then the conversation's third turn: its own checkpoint was kept as well
    assert _run(runner, side, 0)[0] == 0
    c = b + tok.encode(" Which one does Rust's standard library use?")
    assert runner.begin(c) == 0 and runner.reused >= (len(b) - 1) // 32 * 32 - 32
    runner.feed(c)
    assert runner.tokens == c
