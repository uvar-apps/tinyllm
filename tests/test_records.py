from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tinyllm import pin
from tinyllm.bench import record as bench_record
from tinyllm.oracle.record import OracleRecord, PromptRecord, Step, compare


def _bench_doc(decode: float, prefill: float, fingerprint: str = "abc") -> dict[str, Any]:
    cfg = {"rows": 1, "chunk": 32, "mode": "sample", "prefill_tokens": 256, "max_context": 4096}
    return {
        "model": {"file": "m.gguf", "fingerprint": fingerprint},
        "cases": [{"config": cfg, "result": {"decode": {"tok_s": decode}, "prefill": {"tok_s": prefill}}}],
    }


def test_bench_compare_flags_only_slowdowns_beyond_tolerance() -> None:
    deltas, missing = bench_record.compare(_bench_doc(12.5, 190.0), _bench_doc(12.2, 180.0))
    assert not missing
    by_metric = {d.metric: d for d in deltas}
    assert by_metric["decode.tok_s"].change == pytest.approx(-0.024)
    assert not by_metric["decode.tok_s"].regressed(0.03)
    assert by_metric["prefill.tok_s"].regressed(0.03)
    faster, _ = bench_record.compare(_bench_doc(12.5, 190.0), _bench_doc(25.0, 300.0))
    assert not any(d.regressed(0.03) for d in faster)


def test_bench_compare_reports_cases_without_baseline_and_wrong_model() -> None:
    new = _bench_doc(12.5, 190.0)
    new["cases"][0]["config"]["rows"] = 8
    assert bench_record.compare(_bench_doc(12.5, 190.0), new) == (
        [],
        ["rows=8 chunk=32 mode=sample prefill=256 ctx=4096"],
    )
    with pytest.raises(ValueError, match="baseline is for"):
        bench_record.compare(_bench_doc(1, 1, "abc"), _bench_doc(1, 1, "xyz"))


def test_bench_compare_skips_metrics_the_baseline_lacks() -> None:
    new = _bench_doc(12.5, 190.0)
    new["cases"][0]["result"]["prefill"]["pipelined_tok_s"] = 200.0
    deltas, _ = bench_record.compare(_bench_doc(12.5, 190.0), new)
    assert sorted(d.metric for d in deltas) == ["decode.tok_s", "prefill.tok_s"]
    both, _ = bench_record.compare(new, new)
    assert "prefill.pipelined_tok_s" in {d.metric for d in both}


def test_bench_merge_replaces_matching_cases_and_keeps_the_rest() -> None:
    existing = _bench_doc(12.5, 190.0)
    other = copy.deepcopy(existing["cases"][0])
    other["config"]["rows"] = 8
    existing["cases"].append(other)
    merged = bench_record.merge(existing, _bench_doc(13.0, 200.0))
    assert [(c["config"]["rows"], c["result"]["decode"]["tok_s"]) for c in merged["cases"]] == [(8, 12.5), (1, 13.0)]
    with pytest.raises(ValueError, match="different models"):
        bench_record.merge(existing, _bench_doc(1, 1, "xyz"))


def test_bench_document_records_the_model_by_name_not_by_local_path(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + bytes(64))
    case = {"config": {"model": str(model), "rows": 1}, "result": {"tinygrad_rev": "abc", "device": {}}}
    doc = bench_record.document(model, [case])
    assert doc["cases"][0]["config"]["model"] == "m.gguf"
    assert str(tmp_path) not in json.dumps(doc)


def test_host_facts_have_no_identity() -> None:
    facts = bench_record.host_facts()
    assert set(facts) == {"cpu", "memory_gib", "kernel", "python", "power_profile"}


def _oracle(device_kind: str = "cpu") -> OracleRecord:
    steps = [Step([5, 9, 2], [10.0, 8.0, 1.0], 10.2), Step([7, 5, 1], [6.0, 5.5, 0.5], 6.6)]
    prompt = PromptRecord("p", "hello", [1, 2, 3], [5, 7], steps)
    return OracleRecord({"file": "m.gguf", "fingerprint": "abc"}, device_kind, "rev", {"decode": 2}, [prompt])


def test_oracle_roundtrip_and_logprob() -> None:
    again = OracleRecord.from_json(_oracle().to_json())
    assert again == _oracle()
    assert again.prompts[0].steps[0].logprob(5) == pytest.approx(-0.2)
    assert again.prompts[0].steps[0].logprob(42) is None


def test_oracle_identical_records_match() -> None:
    result = compare(_oracle(), _oracle())
    assert result.ok and result.steps == 2 and result.max_logit_diff == 0


def test_oracle_small_logit_drift_is_tolerated_large_is_not() -> None:
    drift = copy.deepcopy(_oracle())
    drift.prompts[0].steps[1].top_logits[0] += 0.01
    assert compare(_oracle(), drift).ok
    drift.prompts[0].steps[1].top_logits[0] += 0.5
    bad = compare(_oracle(), drift)
    assert not bad.ok and bad.logit_violations and bad.max_logit_diff == pytest.approx(0.51)
    # across device kinds the tolerance is loose
    drift.device_kind = "amd"
    assert compare(_oracle(), drift).ok


def test_oracle_token_divergence_stops_the_comparison() -> None:
    other = copy.deepcopy(_oracle())
    other.prompts[0].gen_ids = [9, 7]
    other.prompts[0].steps[0] = Step([9, 5, 2], [10.1, 10.0, 1.0], 10.8)
    result = compare(_oracle(), other)
    assert result.token_mismatches == ["p@0: 5 -> 9"] and result.steps == 1 and not result.ok


def test_oracle_tokenizer_change_is_reported() -> None:
    other = copy.deepcopy(_oracle())
    other.prompts[0].prompt_ids = [1, 2]
    result = compare(_oracle(), other)
    assert result.tokenizer_mismatches and result.steps == 0


def test_installed_tinygrad_is_the_pinned_commit() -> None:
    assert pin.installed_tinygrad_rev() == pin.TINYGRAD_REV
    assert pin.check() == pin.TINYGRAD_REV


def test_pin_mismatch_is_an_error_unless_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pin, "installed_tinygrad_rev", lambda: "0" * 40)
    with pytest.raises(pin.PinError, match="pinned to tinygrad"):
        pin.check()
    monkeypatch.setenv(pin.ENV_UNPINNED, "1")
    assert pin.check() == "0" * 40
