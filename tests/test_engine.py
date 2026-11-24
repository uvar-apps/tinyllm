"""The engine loop with a predictable runner: generation, limits, cancellation, deadlines, reuse, errors."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator

import pytest
from fakes import CountingRunner, DeviceCountingRunner, SpecCountingRunner

from tinyllm.engine.core import Engine, EngineRequest, Event, FinishEvent, TokenEvent
from tinyllm.engine.sampling import SamplingParams

GREEDY = SamplingParams(temperature=0)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = Engine(CountingRunner(delay=0.001), max_waiting=2, prefill_slice=4)
    thread = threading.Thread(target=eng.run, daemon=True)
    thread.start()
    yield eng
    eng.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _collect(events: queue.Queue[Event], timeout: float = 5) -> tuple[list[int], FinishEvent]:
    tokens = []
    while True:
        event = events.get(timeout=timeout)
        if isinstance(event, FinishEvent):
            return tokens, event
        assert isinstance(event, TokenEvent)
        tokens.append(event.token)


def _submit(engine: Engine, request: EngineRequest) -> queue.Queue[Event]:
    events: queue.Queue[Event] = queue.Queue()
    assert engine.submit(request, events.put)
    return events


def test_generates_until_max_tokens(engine: Engine) -> None:
    tokens, done = _collect(_submit(engine, EngineRequest("a", list(range(10)), GREEDY, max_tokens=5)))
    assert tokens == [10, 11, 12, 13, 14]
    assert (done.reason, done.prompt_tokens, done.completion_tokens) == ("length", 10, 5)
    assert done.first_token_seconds is not None and done.queue_seconds >= 0


def test_prompt_is_fed_in_slices(engine: Engine) -> None:
    _collect(_submit(engine, EngineRequest("a", list(range(10)), GREEDY, max_tokens=2)))
    assert engine.runner.calls[:3] == [4, 4, 2] and engine.runner.calls[3:] == [1]  # type: ignore[attr-defined]


def test_end_token_finishes_without_being_emitted(engine: Engine) -> None:
    request = EngineRequest("a", [1, 2], GREEDY, max_tokens=50, end_tokens=frozenset({5}))
    tokens, done = _collect(_submit(engine, request))
    assert tokens == [3, 4] and done.reason == "stop" and done.completion_tokens == 3


def test_queue_limit_refuses_beyond_it(engine: Engine) -> None:
    slow = SamplingParams(temperature=0)
    first = _submit(engine, EngineRequest("a", [1], slow, max_tokens=2000))
    others = [_submit(engine, EngineRequest(f"w{i}", [1], slow, max_tokens=1)) for i in range(2)]
    assert not engine.submit(EngineRequest("x", [1], slow), lambda e: None)
    engine.cancel("a")
    assert _collect(first)[1].reason == "cancelled"
    for events in others:
        assert _collect(events)[1].reason == "length"
    assert engine.submit(EngineRequest("y", [1], slow, max_tokens=1), lambda e: None)  # room again


def test_cancel_while_running_and_while_waiting(engine: Engine) -> None:
    running = _submit(engine, EngineRequest("a", [1], GREEDY, max_tokens=100_000))
    waiting = _submit(engine, EngineRequest("b", [1], GREEDY, max_tokens=5))
    running.get(timeout=5)  # it has started
    engine.cancel("b")
    engine.cancel("a")
    tokens, done = _collect(running)
    assert done.reason == "cancelled" and done.completion_tokens >= 1
    tokens, done = _collect(waiting)
    assert tokens == [] and done.reason == "cancelled" and done.completion_tokens == 0


def test_queue_timeout(engine: Engine) -> None:
    long = _submit(engine, EngineRequest("a", [1], GREEDY, max_tokens=100_000))
    late = _submit(engine, EngineRequest("b", [1], GREEDY, max_tokens=5, queue_timeout=0.05))
    long.get(timeout=5)
    time.sleep(0.2)
    engine.cancel("a")
    assert _collect(late)[1].reason == "timeout"


def test_state_is_reused_when_the_prompt_extends_it(engine: Engine) -> None:
    _tokens, done = _collect(_submit(engine, EngineRequest("a", [1, 2, 3], GREEDY, max_tokens=3)))
    assert done.cached_tokens == 0
    # the state holds 1 2 3 4 5 (the last sampled token, 6, was never fed)
    _, done = _collect(_submit(engine, EngineRequest("b", [1, 2, 3, 4, 5, 6, 7], GREEDY, max_tokens=1)))
    assert done.cached_tokens == 5
    _, done = _collect(_submit(engine, EngineRequest("c", [9, 9], GREEDY, max_tokens=1)))
    assert done.cached_tokens == 0


def test_context_limit_ends_with_length() -> None:
    eng = Engine(CountingRunner(max_context=8), prefill_slice=4)
    thread = threading.Thread(target=eng.run, daemon=True)
    thread.start()
    events: queue.Queue[Event] = queue.Queue()
    eng.submit(EngineRequest("a", [1, 2, 3], GREEDY, max_tokens=100), events.put)
    tokens, done = _collect(events)
    eng.stop()
    thread.join(5)
    assert done.reason == "length" and 3 + len(tokens) <= 8


def test_a_failing_request_does_not_stop_the_engine() -> None:
    eng = Engine(CountingRunner(fail_on=99), prefill_slice=4)
    thread = threading.Thread(target=eng.run, daemon=True)
    thread.start()
    bad: queue.Queue[Event] = queue.Queue()
    good: queue.Queue[Event] = queue.Queue()
    eng.submit(EngineRequest("bad", [99], GREEDY), bad.put)
    eng.submit(EngineRequest("good", [1], GREEDY, max_tokens=2), good.put)
    _, failed = _collect(bad)
    tokens, done = _collect(good)
    eng.stop()
    thread.join(5)
    assert failed.reason == "error" and "device fault" in (failed.error or "")
    assert tokens == [2, 3] and done.reason == "length"
    assert eng.metrics.finished["error"] == 1 and "tinyllm_requests_finished_total" in eng.metrics.render()


def test_late_cancels_are_forgotten(engine: Engine) -> None:
    _collect(_submit(engine, EngineRequest("a", [1], GREEDY, max_tokens=2)))
    engine.cancel("a")  # after it finished, as a client closing late does
    engine.cancel("never-submitted")
    _collect(_submit(engine, EngineRequest("b", [1], GREEDY, max_tokens=2)))
    assert not engine._cancelled


def _run_one(engine: Engine, request: EngineRequest) -> tuple[list[Event], FinishEvent]:
    thread = threading.Thread(target=engine.run, daemon=True)
    thread.start()
    events: queue.Queue[Event] = queue.Queue()
    engine.submit(request, events.put)
    out = []
    while not isinstance(event := events.get(timeout=5), FinishEvent):
        out.append(event)
    engine.stop()
    thread.join(5)
    return out, event


def test_device_sampling_when_the_settings_allow() -> None:
    runner = DeviceCountingRunner()
    events, done = _run_one(Engine(runner), EngineRequest("a", [1, 2], GREEDY, max_tokens=5))
    assert [e.token for e in events if isinstance(e, TokenEvent)] == [3, 4, 5, 6, 7] and done.reason == "length"
    assert runner.sampled_calls == 4  # every token after the first (which comes from the prefill's logits)
    runner = DeviceCountingRunner()
    penalised = SamplingParams(temperature=0, presence_penalty=0.5)
    _run_one(Engine(runner), EngineRequest("b", [1, 2], penalised, max_tokens=5))
    assert runner.sampled_calls == 0  # penalties need the whole row: host path
    runner = DeviceCountingRunner()
    _run_one(Engine(runner, device_sampling=False), EngineRequest("c", [1, 2], GREEDY, max_tokens=5))
    assert runner.sampled_calls == 0


def test_device_logprobs_match_the_host_path() -> None:
    params = SamplingParams(temperature=0, logprobs=True, top_logprobs=3)
    device_events, _ = _run_one(Engine(DeviceCountingRunner()), EngineRequest("a", [1, 2], params, max_tokens=4))
    host_events, _ = _run_one(Engine(CountingRunner()), EngineRequest("b", [1, 2], params, max_tokens=4))
    for d, h in zip(device_events, host_events, strict=True):
        assert isinstance(d, TokenEvent) and isinstance(h, TokenEvent) and d.token == h.token
        assert d.sample.logprob == pytest.approx(h.sample.logprob, abs=1e-4)
        assert [t for t, _ in d.sample.top] == [t for t, _ in h.sample.top]


def _tokens(events: list[Event]) -> list[int]:
    return [e.token for e in events if isinstance(e, TokenEvent)]


def test_speculative_cycles_give_the_plain_output() -> None:
    plain, _ = _run_one(Engine(DeviceCountingRunner()), EngineRequest("a", [1, 2], GREEDY, max_tokens=12))
    runner = SpecCountingRunner(speculative=3)
    events, done = _run_one(Engine(runner), EngineRequest("b", [1, 2], GREEDY, max_tokens=12))
    assert _tokens(events) == _tokens(plain) and done.reason == "length" and done.completion_tokens == 12
    assert runner.cycles[0] == 3 and runner.sampled_calls == 0  # every draft holds: 4 tokens per cycle
    # rejected drafts cost tokens per cycle, never correctness
    runner = SpecCountingRunner(speculative=3, wrong=frozenset({4, 9, 10}))
    events, _ = _run_one(Engine(runner), EngineRequest("c", [1, 2], GREEDY, max_tokens=12))
    assert _tokens(events) == _tokens(plain) and min(runner.cycles) < 3
    assert runner.tokens[:2] == [1, 2] and runner.tokens[2:] == _tokens(events)[: len(runner.tokens) - 2]


def test_speculative_stops_inside_a_cycle() -> None:
    runner = SpecCountingRunner(speculative=3)
    request = EngineRequest("a", [1, 2], GREEDY, max_tokens=50, end_tokens=frozenset({6}))
    events, done = _run_one(Engine(runner), request)
    assert _tokens(events) == [3, 4, 5] and done.reason == "stop"
    events, done = _run_one(Engine(SpecCountingRunner(speculative=3)), EngineRequest("b", [1, 2], GREEDY, max_tokens=6))
    assert _tokens(events) == [3, 4, 5, 6, 7, 8] and done.reason == "length" and done.completion_tokens == 6


def test_speculative_seeded_sampling_matches_plain_decoding() -> None:
    params = SamplingParams(temperature=40.0, top_k=6, seed=11)  # flat enough that the draws vary
    plain, _ = _run_one(Engine(DeviceCountingRunner()), EngineRequest("a", [1, 2], params, max_tokens=40))
    runner = SpecCountingRunner(speculative=3)
    spec, _ = _run_one(Engine(runner), EngineRequest("b", [1, 2], params, max_tokens=40))
    assert _tokens(spec) == _tokens(plain)
    assert len(set(_tokens(plain))) > 5 and 0 in runner.cycles  # draws differ from the drafts and get rejected


def test_speculative_off_near_the_context_limit_and_when_disabled() -> None:
    runner = SpecCountingRunner(speculative=3, max_context=10)
    events, done = _run_one(Engine(runner), EngineRequest("a", [1, 2], GREEDY, max_tokens=20))
    assert _tokens(events) == list(range(3, 3 + len(_tokens(events)))) and done.reason == "length"
    assert runner.sampled_calls > 0  # the last positions decode one token at a time
    runner = SpecCountingRunner(speculative=3)
    _run_one(Engine(runner, speculative=False), EngineRequest("b", [1, 2], GREEDY, max_tokens=5))
    assert not runner.cycles and runner.sampled_calls == 4


def test_prompt_slices_carry_the_next_prompt_token() -> None:
    runner = CountingRunner()
    _run_one(Engine(runner, prefill_slice=4), EngineRequest("a", list(range(10)), GREEDY, max_tokens=2))
    assert runner.followings[:3] == [4, 8, None]


def test_latency_and_decode_metrics(engine: Engine) -> None:
    _collect(_submit(engine, EngineRequest("a", list(range(10)), GREEDY, max_tokens=5)))
    m = engine.metrics
    assert m.e2e_time.n == m.prefill_time.n == m.decode_time.n == m.time_per_output_token.n == 1
    assert m.inter_token_latency.n == 4  # five streamed tokens, four gaps between them
    assert m.time_per_output_token.total == pytest.approx(m.decode_time.total / 4)
    assert 0 < m.prefill_time.total <= m.e2e_time.total
    text = m.render()
    for name in (
        "e2e_request_latency_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "time_per_output_token_seconds",
        "inter_token_latency_seconds",
    ):
        assert f"tinyllm_{name}_count 1" in text or f"tinyllm_{name}_count 4" in text, name
    assert 'tinyllm_inter_token_latency_seconds_bucket{le="+Inf"} 4' in text


def test_a_request_that_never_started_records_only_its_wait(engine: Engine) -> None:
    long = _submit(engine, EngineRequest("a", [1], GREEDY, max_tokens=100_000))
    late = _submit(engine, EngineRequest("b", [1], GREEDY, max_tokens=5, queue_timeout=0.05))
    long.get(timeout=5)
    time.sleep(0.2)
    engine.cancel("a")
    assert _collect(late)[1].reason == "timeout"
    _collect(long)
    m = engine.metrics
    assert m.e2e_time.n == 2 and m.prefill_time.n == 1 and m.decode_time.n == 1  # only "a" ran


def test_histograms_take_their_own_buckets() -> None:
    from tinyllm.engine.metrics import TOKEN_BUCKETS, Histogram

    h = Histogram(TOKEN_BUCKETS)
    for value in (0.0005, 0.04, 5.0):
        h.observe(value)
    assert len(h.counts) == len(TOKEN_BUCKETS) + 1 and h.counts[0] == 1 and h.counts[-1] == 1 and h.n == 3
