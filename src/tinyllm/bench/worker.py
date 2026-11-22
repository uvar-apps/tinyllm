"""Runs one benchmark case in its own process and prints the result as a JSON line.

A fresh process per case is required: tinygrad's model fixes its batch size and compiles its
programs on first use, and a clean process is also what a server start looks like.

    python -m tinyllm.bench.worker '{"model": "...", "rows": 1, ...}'
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Any

RESULT_PREFIX = "TINYLLM_BENCH_RESULT "
WARMUP_CALLS = 3  # tinygrad's JIT runs twice before replaying; the first replay also allocates


@dataclass(frozen=True)
class Case:
    model: str
    device: str | None = None
    rows: int = 1
    chunk: int = 32
    mode: str = "sample"  # "sample" or "logits"
    prefill_tokens: int = 256
    decode_steps: int = 32
    max_context: int = 4096

    def key(self) -> str:
        return f"rows={self.rows} chunk={self.chunk} mode={self.mode}"


def _summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "median_ms": statistics.median(ordered) * 1e3,
        "min_ms": ordered[0] * 1e3,
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] * 1e3,
        "samples": len(ordered),
    }


def run(case: Case) -> dict[str, Any]:
    from tinyllm import device

    device.configure(case.device)
    import numpy as np

    from tinyllm import tg
    from tinyllm.tg.upstream import UpstreamSession

    if case.mode not in ("sample", "logits"):
        raise ValueError(f"unknown mode {case.mode!r}")
    if case.prefill_tokens // case.chunk < WARMUP_CALLS + 2 or case.decode_steps < WARMUP_CALLS + 5:
        raise ValueError("too few prefill chunks or decode steps to measure past warm-up")
    if 2 * case.prefill_tokens + case.decode_steps >= case.max_context:
        raise ValueError("the context must hold the prompt twice plus the decode steps")

    logits = case.mode == "logits"
    session = UpstreamSession(case.model, case.max_context, case.rows, case.chunk, logits=logits)
    # deterministic synthetic prompt: token ids only, so tokenizer speed stays out of the measurement
    prompt = [1000 + i % 1000 for i in range(case.prefill_tokens)]
    out = session.prefill(prompt)
    for _ in range(case.decode_steps):
        out = session.step([int(i) for i in np.argmax(out, axis=-1)]) if logits else session.step()
    # the same number of tokens again, queued without waiting between calls
    session.prefill(prompt, wait_each=False)

    prefill = [r for r in session.trace if r.kind == "prefill"]
    decode = [r for r in session.trace if r.kind == "decode"]
    full = [r for r in prefill[WARMUP_CALLS:] if r.tokens == case.chunk]
    warm_decode = decode[WARMUP_CALLS:]
    pipelined = next(r for r in session.trace if r.kind == "prefill_pipelined")
    p, d = _summary([r.seconds for r in full]), _summary([r.seconds for r in warm_decode])
    decode_bytes = statistics.median(r.bytes_touched for r in warm_decode)
    return {
        "load_s": session.load_seconds,
        "warmup_s": sum(r.seconds for r in prefill[:WARMUP_CALLS]) + sum(r.seconds for r in decode[:WARMUP_CALLS]),
        "prefill": {
            **p,
            "tok_s": case.rows * case.chunk / (p["median_ms"] / 1e3),
            # calls queued back to back, as a single-stream prefill loop runs them
            "pipelined_tok_s": case.rows * pipelined.tokens / pipelined.seconds,
            "pipelined_ms_per_call": pipelined.seconds / pipelined.calls * 1e3,
        },
        "decode": {
            **d,
            "tok_s": case.rows / (d["median_ms"] / 1e3),
            "tok_s_per_row": 1 / (d["median_ms"] / 1e3),
            "gb_per_step": decode_bytes / 1e9,
            "gb_s": decode_bytes / 1e9 / (d["median_ms"] / 1e3),
        },
        "device": tg.device_info(),
        "tinygrad_rev": tg.TINYGRAD_REV,
    }


def main() -> int:
    case = Case(**json.loads(sys.argv[1]))
    result = run(case)
    print(RESULT_PREFIX + json.dumps({"config": asdict(case), "result": result}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
