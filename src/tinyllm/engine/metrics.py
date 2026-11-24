"""Request metrics, rendered in Prometheus' text format at `/metrics` (design 4.13, first set).

Updated on the engine thread and read from the server thread; each update is a few attribute writes, and a
reader sees at worst a slightly stale value.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

# upper bounds in seconds; the last bucket is +Inf
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
# whole requests and their decode phase: long answers with thinking run for many minutes
REQUEST_BUCKETS = (0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0)
# per token: 8 to 60 tok/s while decoding; a speculative step streams its tokens together, a millisecond apart
TOKEN_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)


@dataclass
class Histogram:
    buckets: tuple[float, ...] = LATENCY_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    n: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.n += 1
        self.total += value
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def render(self, name: str, help_text: str) -> list[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        running = 0
        for bound, count in zip((*self.buckets, float("inf")), self.counts, strict=True):
            running += count
            label = "+Inf" if bound == float("inf") else f"{bound:g}"
            lines.append(f'{name}_bucket{{le="{label}"}} {running}')
        return [*lines, f"{name}_sum {self.total:.6f}", f"{name}_count {self.n}"]


@dataclass
class Metrics:
    running: int = 0
    waiting: int = 0
    finished: collections.Counter[str] = field(default_factory=collections.Counter)
    prompt_tokens: int = 0
    generation_tokens: int = 0
    cached_tokens: int = 0
    queue_time: Histogram = field(default_factory=Histogram)
    first_token_time: Histogram = field(default_factory=Histogram)
    e2e_time: Histogram = field(default_factory=lambda: Histogram(REQUEST_BUCKETS))
    prefill_time: Histogram = field(default_factory=Histogram)
    decode_time: Histogram = field(default_factory=lambda: Histogram(REQUEST_BUCKETS))
    time_per_output_token: Histogram = field(default_factory=lambda: Histogram(TOKEN_BUCKETS))
    inter_token_latency: Histogram = field(default_factory=lambda: Histogram(TOKEN_BUCKETS))
    spec_cycles: int = 0  # draft-and-verify cycles
    spec_draft_tokens: int = 0  # drafts proposed
    spec_accepted_tokens: int = 0  # drafts the main model agreed with

    def record_cycle(self, drafts: int, accepted: int) -> None:
        self.spec_cycles += 1
        self.spec_draft_tokens += drafts
        self.spec_accepted_tokens += accepted

    def record_token_gap(self, seconds: float) -> None:
        """The time between two tokens streamed to a client."""
        self.inter_token_latency.observe(seconds)

    def record_finish(
        self,
        reason: str,
        prompt: int,
        generated: int,
        cached: int,
        queued: float,
        first_token: float | None,
        e2e: float | None = None,
        prefill: float | None = None,
        decode: float | None = None,
    ) -> None:
        """One finished request. `e2e`: arrival to finish; `prefill`: time feeding the prompt (None when nothing
        was fed); `decode`: first token to finish, over which `generated - 1` more tokens came."""
        self.finished[reason] += 1
        self.prompt_tokens += prompt
        self.generation_tokens += generated
        self.cached_tokens += cached
        self.queue_time.observe(queued)
        if first_token is not None:
            self.first_token_time.observe(first_token)
        if e2e is not None:
            self.e2e_time.observe(e2e)
        if prefill is not None:
            self.prefill_time.observe(prefill)
        if decode is not None and generated > 1:
            self.decode_time.observe(decode)
            self.time_per_output_token.observe(decode / (generated - 1))

    def render(self) -> str:
        p = "tinyllm"
        lines = [
            f"# HELP {p}_requests_running Requests being processed.",
            f"# TYPE {p}_requests_running gauge",
            f"{p}_requests_running {self.running}",
            f"# HELP {p}_requests_waiting Requests waiting to start.",
            f"# TYPE {p}_requests_waiting gauge",
            f"{p}_requests_waiting {self.waiting}",
            f"# HELP {p}_requests_finished_total Finished requests by reason.",
            f"# TYPE {p}_requests_finished_total counter",
            *(f'{p}_requests_finished_total{{reason="{r}"}} {n}' for r, n in sorted(self.finished.items())),
        ]
        for name, value, text in (
            ("prompt_tokens_total", self.prompt_tokens, "Prompt tokens of finished requests."),
            ("generation_tokens_total", self.generation_tokens, "Generated tokens of finished requests."),
            ("prefix_cache_hit_tokens_total", self.cached_tokens, "Prompt tokens served from reused state."),
            ("spec_decode_num_drafts_total", self.spec_cycles, "Speculative draft-and-verify cycles."),
            ("spec_decode_num_draft_tokens_total", self.spec_draft_tokens, "Tokens drafted (MTP or DFlash2)."),
            ("spec_decode_num_accepted_tokens_total", self.spec_accepted_tokens, "Drafted tokens accepted."),
        ):
            lines += [f"# HELP {p}_{name} {text}", f"# TYPE {p}_{name} counter", f"{p}_{name} {value}"]
        lines += self.queue_time.render(f"{p}_queue_time_seconds", "Time from arrival to start.")
        lines += self.first_token_time.render(f"{p}_time_to_first_token_seconds", "Time from arrival to first token.")
        lines += self.e2e_time.render(f"{p}_e2e_request_latency_seconds", "Time from arrival to finish.")
        lines += self.prefill_time.render(f"{p}_request_prefill_time_seconds", "Time feeding the prompt.")
        lines += self.decode_time.render(f"{p}_request_decode_time_seconds", "Time from first token to finish.")
        lines += self.time_per_output_token.render(
            f"{p}_time_per_output_token_seconds", "Decode time per token after the first, per request."
        )
        lines += self.inter_token_latency.render(
            f"{p}_inter_token_latency_seconds", "Time between two streamed tokens."
        )
        return "\n".join(lines) + "\n"
