"""Benchmark records: host facts, the document layout, and comparison against a baseline."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinyllm import __version__
from tinyllm.weights.resolve import fingerprint

SCHEMA = 1
# (section, field): higher is better for all of them
METRICS = (("decode", "tok_s"), ("prefill", "tok_s"), ("prefill", "pipelined_tok_s"))


def _first_line(path: str, prefix: str) -> str | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def host_facts() -> dict[str, Any]:
    """Hardware and OS facts that explain a number. No host name or user name."""
    profile: str | None
    try:
        profile = (
            subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        profile = None
    mem_kb = _first_line("/proc/meminfo", "MemTotal")
    return {
        "cpu": _first_line("/proc/cpuinfo", "model name"),
        "memory_gib": round(int(mem_kb.split()[0]) / (1 << 20), 1) if mem_kb else None,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "power_profile": profile,
    }


def model_facts(path: Path) -> dict[str, Any]:
    return {"file": path.name, "bytes": path.stat().st_size, "fingerprint": fingerprint(path)}


def _portable(case: dict[str, Any]) -> dict[str, Any]:
    """A case without machine-specific detail: the model is recorded by file name, not by local path."""
    return {**case, "config": {**case["config"], "model": Path(case["config"]["model"]).name}}


def document(model: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [_portable(c) for c in cases]
    first = cases[0]["result"] if cases else {}
    return {
        "schema": SCHEMA,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "tinyllm": __version__,
        "tinygrad_rev": first.get("tinygrad_rev"),
        "host": host_facts(),
        "device": first.get("device"),
        "model": model_facts(model),
        "cases": cases,
    }


def case_key(config: dict[str, Any]) -> str:
    return (
        f"rows={config['rows']} chunk={config['chunk']} mode={config['mode']} "
        f"prefill={config['prefill_tokens']} ctx={config['max_context']}"
    )


def merge(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Add the cases of `new` to an existing record of the same model, replacing cases with the same key."""
    if existing["model"]["fingerprint"] != new["model"]["fingerprint"]:
        raise ValueError(f"{existing['model']['file']} and {new['model']['file']} are different models")
    fresh = {case_key(c["config"]) for c in new["cases"]}
    kept = [c for c in existing["cases"] if case_key(c["config"]) not in fresh]
    return {**new, "cases": kept + new["cases"]}


@dataclass(frozen=True)
class Delta:
    case: str
    metric: str
    base: float
    new: float

    @property
    def change(self) -> float:
        return self.new / self.base - 1 if self.base else 0.0

    def regressed(self, tolerance: float) -> bool:
        return self.change < -tolerance


def compare(base: dict[str, Any], new: dict[str, Any]) -> tuple[list[Delta], list[str]]:
    """Deltas for every case present in both records, plus the cases the baseline lacks."""
    if base["model"]["fingerprint"] != new["model"]["fingerprint"]:
        raise ValueError(f"baseline is for {base['model']['file']}, not {new['model']['file']}")
    base_cases = {case_key(c["config"]): c["result"] for c in base["cases"]}
    deltas: list[Delta] = []
    missing: list[str] = []
    for c in new["cases"]:
        key = case_key(c["config"])
        if key not in base_cases:
            missing.append(key)
            continue
        for section, field in METRICS:
            old, cur = base_cases[key][section].get(field), c["result"][section].get(field)
            if old is not None and cur is not None:  # records written before a metric existed lack it
                deltas.append(Delta(key, f"{section}.{field}", old, cur))
    return deltas, missing
