"""The engine loop (design 4.3, phase 1: one sequence at a time).

`Engine.run` owns the main thread and the runner (tinygrad's devices and JIT are not thread-safe, decision D5).
Other threads talk to it through `submit` and `cancel`, which only put messages on a queue, and receive each
request's output through the `sink` they passed in, called on the engine thread.

A request waits in a FIFO queue (at most `max_waiting` of them; beyond that `submit` refuses, which the server
maps to HTTP 429) until the running one finishes. Its prompt is fed in slices of a few prefill chunks, so a
cancellation is seen within one slice; then one token per step. Cancellation and queue deadlines are checked
between steps.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tinyllm.engine.metrics import Metrics
from tinyllm.engine.sampling import Sample, Sampler, SamplingParams
from tinyllm.runner.base import Runner

log = logging.getLogger("tinyllm.engine")


@dataclass
class EngineRequest:
    id: str
    prompt: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    max_tokens: int = 256
    end_tokens: frozenset[int] = frozenset()
    queue_timeout: float | None = None  # seconds it may wait before it starts; None waits as long as it takes
    arrival: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class TokenEvent:
    token: int
    sample: Sample


@dataclass(frozen=True)
class FinishEvent:
    reason: str  # "stop", "length", "cancelled", "timeout" or "error"
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    error: str | None = None
    queue_seconds: float = 0.0
    first_token_seconds: float | None = None  # arrival to first token
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None


Event = TokenEvent | FinishEvent
Sink = Callable[[Event], None]


@dataclass
class _Active:
    request: EngineRequest
    sink: Sink
    sampler: Sampler
    start: float  # when it left the queue
    cached: int = 0  # prompt tokens whose state was reused
    fed: int = 0  # prompt tokens in the model state
    generated: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None  # when the last token went to the client
    prefill_seconds: float = 0.0
    last: int | None = None  # the last sampled token, still to be fed
    device: bool = False  # decode samples on the device (runner.feed_sampled)
    copied: int = 0  # tokens that came from blocks copied from earlier text (runner.last_drafted)


class Engine:
    def __init__(
        self,
        runner: Runner,
        max_waiting: int = 16,
        prefill_slice: int = 512,
        device_sampling: bool = True,
        speculative: bool = True,
    ) -> None:
        self.runner = runner
        self.device_sampling = device_sampling  # sample on the device when the runner can and the settings allow
        # draft-and-verify cycles (tinyllm.tg.speculative) for device-sampled requests, when the runner has them
        self.speculative = int(getattr(runner, "speculative", 0)) if speculative else 0
        # rows of sampler settings a cycle takes: more than K + 1 when the runner also verifies copied blocks
        self.speculative_rows = int(getattr(runner, "speculative_rows", self.speculative + 1))
        self.max_waiting = max_waiting
        self.prefill_slice = prefill_slice  # prompt tokens per engine step; a multiple of the runner's chunk
        self.metrics = Metrics()
        self._inbox: queue.Queue[tuple[str, object]] = queue.Queue()
        self._waiting: collections.deque[tuple[EngineRequest, Sink]] = collections.deque()
        self._cancelled: set[str] = set()
        self._active: _Active | None = None
        self._lock = threading.Lock()
        self._admitted = 0  # requests submitted and not yet finished, for the queue limit
        self._stopping = False

    # ---- called from any thread -------------------------------------------------------------------------------

    def submit(self, request: EngineRequest, sink: Sink) -> bool:
        """Queue a request; False when the queue is full (nothing was queued)."""
        with self._lock:
            if self._admitted >= self.max_waiting + 1:  # the running request plus the waiting ones
                return False
            self._admitted += 1
        self._inbox.put(("submit", (request, sink)))
        return True

    def cancel(self, request_id: str) -> None:
        self._inbox.put(("cancel", request_id))

    def stop(self) -> None:
        self._inbox.put(("stop", None))

    @property
    def load(self) -> tuple[int, int]:
        """(running, waiting) requests, approximately."""
        with self._lock:
            running = 1 if self._active is not None else 0
            return running, max(0, self._admitted - running)

    # ---- the engine thread -------------------------------------------------------------------------------------

    def run(self) -> None:
        """Serve until `stop` is called."""
        while not self._stopping:
            self._drain(block=self._active is None and not self._waiting)
            if self._stopping:
                break
            if self._active is None:
                self._admit()
            if self._active is not None:
                self.step()
        for request, sink in list(self._waiting):
            self._finish_waiting(request, sink, "cancelled")
        if self._active is not None:
            self._finish(self._active, "cancelled")

    def _drain(self, block: bool) -> None:
        try:
            kind, payload = self._inbox.get(timeout=0.5) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            if kind == "submit":
                self._waiting.append(payload)  # type: ignore[arg-type]
                self.metrics.waiting = len(self._waiting)
            elif kind == "cancel" and self._known(str(payload)):  # a late cancel of a finished request is dropped
                self._cancelled.add(str(payload))
            elif kind == "stop":
                self._stopping = True
            try:
                kind, payload = self._inbox.get_nowait()
            except queue.Empty:
                return

    def _known(self, request_id: str) -> bool:
        active = self._active is not None and self._active.request.id == request_id
        return active or any(r.id == request_id for r, _ in self._waiting)

    def _admit(self) -> None:
        while self._waiting:
            request, sink = self._waiting.popleft()
            self.metrics.waiting = len(self._waiting)
            now = time.monotonic()
            if request.id in self._cancelled:
                self._finish_waiting(request, sink, "cancelled")
            elif request.queue_timeout is not None and now - request.arrival > request.queue_timeout:
                self._finish_waiting(request, sink, "timeout")
            else:
                cached = self.runner.begin(request.prompt)
                sampler = Sampler(request.params, request.prompt)
                device = (
                    self.device_sampling and request.params.device_ok and getattr(self.runner, "device_sampling", False)
                )
                # `cached` is where the prompt is fed from; a runner that also skips later stretches says how
                # many tokens it reuses in all
                reused = getattr(self.runner, "reused", cached)
                self._active = _Active(request, sink, sampler, now, cached=reused, fed=cached, device=device)
                self.metrics.running = 1
                return

    def step(self) -> None:
        """One unit of work for the running request: a prompt slice, or one generated token."""
        a = self._active
        assert a is not None
        r = a.request
        if r.id in self._cancelled:
            self._finish(a, "cancelled")
            return
        try:
            if a.fed < len(r.prompt):
                start = time.monotonic()
                piece = r.prompt[a.fed : a.fed + self.prefill_slice]
                end = a.fed + len(piece)
                logits = self.runner.feed(piece, r.prompt[end] if end < len(r.prompt) else None)
                a.fed += len(piece)
                a.prefill_seconds += time.monotonic() - start
                if a.fed < len(r.prompt):
                    return
            else:
                assert a.last is not None
                if a.fed + 1 >= self.runner.max_context:
                    self._finish(a, "length")
                    return
                k = self.speculative
                if a.device and k and a.fed + self.speculative_rows < self.runner.max_context:
                    # one draft-and-verify cycle: the last token and the drafts the main model agreed with go in,
                    # and the main model's own tokens at those rows come out (n accepted drafts and one more)
                    rows, n = self.runner.feed_speculative(a.last, a.sampler.device_rows(self.speculative_rows))
                    a.fed += n + 1
                    drafted = int(getattr(self.runner, "last_drafted", k))  # K, or a copied block's drafts
                    self.metrics.record_cycle(drafted, n)
                    if drafted != k:
                        a.copied += n
                    for j in range(n + 1):
                        self._emit(a, a.sampler.from_device(rows[j]))
                        if self._active is not a:  # finished: an end token, max_tokens or the context
                            break
                    return
                a.fed += 1
                if a.device:  # sampled inside the decode program: a few candidates come back, not the logits
                    out = self.runner.feed_sampled(a.last, a.sampler.device_params())
                    sample = a.sampler.from_device(out)
                else:
                    sample = a.sampler.sample(self.runner.feed([a.last]))
                self._emit(a, sample)
                return
            sample = a.sampler.sample(logits)
        except Exception as e:  # a failing request must not take the server down
            log.exception("request %s failed", r.id)
            self._finish(a, "error", error=f"{type(e).__name__}: {e}")
            return
        self._emit(a, sample)

    def _emit(self, a: _Active, sample: Sample) -> None:
        r = a.request
        a.generated += 1
        a.last = sample.token
        if a.first_token_at is None:
            a.first_token_at = time.monotonic()
        if sample.token in r.end_tokens:
            self._finish(a, "stop")
            return
        now = time.monotonic()
        if a.last_token_at is not None:
            self.metrics.record_token_gap(now - a.last_token_at)
        a.last_token_at = now
        a.sink(TokenEvent(sample.token, sample))
        if a.generated >= r.max_tokens or a.fed + 1 >= self.runner.max_context:
            self._finish(a, "length")

    def _finish_waiting(self, request: EngineRequest, sink: Sink, reason: str) -> None:
        with self._lock:
            self._admitted -= 1
        self._cancelled.discard(request.id)
        queued = time.monotonic() - request.arrival
        self.metrics.record_finish(reason, len(request.prompt), 0, 0, queued, None, e2e=queued)
        sink(FinishEvent(reason, len(request.prompt), 0, queue_seconds=queued))

    def _finish(self, a: _Active, reason: str, error: str | None = None) -> None:
        r, now = a.request, time.monotonic()
        prompt_fed = max(0, min(a.fed, len(r.prompt)) - a.cached)
        decode_seconds = now - a.first_token_at if a.first_token_at is not None else 0.0
        event = FinishEvent(
            reason,
            len(r.prompt),
            a.generated,
            a.cached,
            error,
            queue_seconds=a.start - r.arrival,
            first_token_seconds=a.first_token_at - r.arrival if a.first_token_at is not None else None,
            prefill_tokens_per_second=prompt_fed / a.prefill_seconds if a.prefill_seconds > 0 else None,
            decode_tokens_per_second=(a.generated - 1) / decode_seconds if a.generated > 1 and decode_seconds else None,
        )
        self._active = None
        self._cancelled.discard(r.id)
        with self._lock:
            self._admitted -= 1
        self.metrics.running = 0
        self.metrics.record_finish(
            reason,
            len(r.prompt),
            a.generated,
            a.cached,
            event.queue_seconds,
            event.first_token_seconds,
            e2e=now - r.arrival,
            prefill=a.prefill_seconds if prompt_fed > 0 else None,
            decode=decode_seconds if a.first_token_at is not None else None,
        )
        log.info(
            "%s %s prompt %d (cached %d) out %d%s  queue %.2fs  ttft %s  prefill %s tok/s  decode %s tok/s",
            r.id,
            reason,
            len(r.prompt),
            a.cached,
            a.generated,
            f" (copied {a.copied})" if a.copied else "",
            event.queue_seconds,
            f"{event.first_token_seconds:.2f}s" if event.first_token_seconds is not None else "-",
            f"{event.prefill_tokens_per_second:.0f}" if event.prefill_tokens_per_second else "-",
            f"{event.decode_tokens_per_second:.1f}" if event.decode_tokens_per_second else "-",
        )
        a.sink(event)
