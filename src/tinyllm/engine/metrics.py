"""Request metrics, rendered in Prometheus' text format at `/metrics` (design 4.13, first set).

Updated on the engine thread and read from the server thread; each update is a few attribute writes, and a
reader sees at worst a slightly stale value.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

# upper bounds in seconds; the last bucket is +Inf
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)


@dataclass
class Histogram:
    buckets: tuple[float, ...] = LATENCY_BUCKETS
    counts: list[int] = field(default_factory=lambda: [0] * (len(LATENCY_BUCKETS) + 1))
    total: float = 0.0
    n: int = 0

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

    def record_finish(
        self, reason: str, prompt: int, generated: int, cached: int, queued: float, first_token: float | None
    ) -> None:
        self.finished[reason] += 1
        self.prompt_tokens += prompt
        self.generation_tokens += generated
        self.cached_tokens += cached
        self.queue_time.observe(queued)
        if first_token is not None:
            self.first_token_time.observe(first_token)

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
        ):
            lines += [f"# HELP {p}_{name} {text}", f"# TYPE {p}_{name} counter", f"{p}_{name} {value}"]
        lines += self.queue_time.render(f"{p}_queue_time_seconds", "Time from arrival to start.")
        lines += self.first_token_time.render(f"{p}_time_to_first_token_seconds", "Time from arrival to first token.")
        return "\n".join(lines) + "\n"
