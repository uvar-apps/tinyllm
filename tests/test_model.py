"""Tests that run the development model on the CPU through tinygrad's own engine."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import GOLDEN_DIR, REPO

from tinyllm.oracle.record import OracleRecord, compare

CPU_GOLDEN = GOLDEN_DIR / "qwen35-0.8b-q8_0.cpu.json"
PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def logits_session(dev_model: Path) -> Any:
    from tinyllm.tg.upstream import UpstreamSession

    return UpstreamSession(dev_model, max_context=1024, rows=1, chunk=32, logits=True)


def _greedy(session: Any, ids: list[int], n: int) -> list[int]:
    session.reset()
    out = session.prefill(ids)
    tokens: list[int] = []
    for _ in range(n):
        tokens.append(int(out[0].argmax()) if session.logits else int(out[0]))
        out = session.step([tokens[-1]])
    return tokens


@pytest.mark.model
def test_runs_on_cpu_with_the_pinned_tinygrad() -> None:
    from tinyllm import pin, tg

    assert tg.default_device() == "CPU" and tg.DEV == "CPU"
    assert tg.TINYGRAD_REV == pin.TINYGRAD_REV


@pytest.mark.model
def test_logits_mode_answers_and_traces(logits_session: Any) -> None:
    tok = logits_session.tokenizer()
    ids = tok.encode(PROMPT)
    tokens = _greedy(logits_session, ids, 4)
    assert tok.decode(tokens).startswith(" Paris")
    kinds = [(r.kind, r.tokens) for r in logits_session.trace[-5:]]
    assert kinds == [("prefill", len(ids)), ("decode", 1), ("decode", 1), ("decode", 1), ("decode", 1)]
    assert logits_session.pos == len(ids) + 4


@pytest.mark.model
def test_restarting_at_position_zero_reproduces_the_output(logits_session: Any) -> None:
    # the recurrent layers carry state between calls; a new sequence must not see the previous one's
    ids = logits_session.tokenizer().encode(PROMPT)
    first = _greedy(logits_session, ids, 6)
    _greedy(logits_session, logits_session.tokenizer().encode("def fibonacci(n):"), 6)
    assert _greedy(logits_session, ids, 6) == first


@pytest.mark.model
def test_logits_mode_matches_upstream_sampler(dev_model: Path, logits_session: Any) -> None:
    """Our copy of `Transformer.forward` must stay equal to upstream's: same greedy tokens."""
    from tinyllm.tg.upstream import UpstreamSession

    ids = logits_session.tokenizer().encode(PROMPT)
    upstream = UpstreamSession(dev_model, max_context=1024, rows=1, chunk=32, logits=False)
    assert _greedy(upstream, ids, 8) == _greedy(logits_session, ids, 8)


@pytest.mark.model
def test_cpu_golden(dev_model: Path, logits_session: Any) -> None:
    """Outputs of the pinned tinygrad on the CPU. A failure after a tinygrad bump means upstream changed behaviour."""
    from tinyllm.oracle.capture import capture_session

    golden = OracleRecord.load(CPU_GOLDEN)
    prompts = [{"id": p.id, "text": p.text} for p in golden.prompts]
    fresh = capture_session(logits_session, dev_model, prompts, golden.config["decode"], golden.config["topk"])
    result = compare(golden, fresh)
    assert result.ok, result.summary()


@pytest.mark.model
@pytest.mark.reference
def test_llamacpp_agrees_with_cpu_golden(dev_model: Path, reference_server: Path) -> None:
    from tinyllm.oracle import llamacpp

    golden = OracleRecord.load(CPU_GOLDEN)
    with llamacpp.ReferenceServer(dev_model, reference_server, golden.config["max_context"], threads=8) as server:
        report = llamacpp.compare(golden, server, golden.config["topk"])
    assert not report.tokenizer_mismatches, report.summary()
    assert report.top1_rate >= 0.9, report.summary()


@pytest.mark.gpu
@pytest.mark.parametrize("golden", sorted(GOLDEN_DIR.glob("*.amd.json")), ids=lambda p: p.name)
def test_amd_golden(golden: Path) -> None:
    """Runs in a subprocess: this test process is fixed to the CPU."""
    cmd = [sys.executable, "-m", "tinyllm.cli", "oracle", "check", str(golden), "--device", "amd"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=900, env=_gpu_env())
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]


@pytest.mark.gpu
def test_fixed_length_chunks_match_variable_length() -> None:
    """Full prefill chunks through their own program, the partial last one through the shared one (2.5 chunks)."""
    cmd = [sys.executable, "tools/static_chunk_check.py", "Qwen3.5-0.8B-UD-Q4_K_XL.gguf", "--chunk", "16"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=900, env=_gpu_env())
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert proc.stdout.count("same greedy token: True") == 5


def _gpu_env() -> dict[str, str]:
    import os

    env = {k: v for k, v in os.environ.items() if k not in ("TINYLLM_DEVICE", "DEV")}
    return env
