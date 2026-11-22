"""`tinyllm bench`: run cases in subprocesses, print a table, write and compare records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinyllm.bench import record
from tinyllm.bench.worker import RESULT_PREFIX, Case
from tinyllm.weights.resolve import find_model


def run_case(case: Case, timeout: float = 3600) -> dict[str, Any]:
    """Run one case in a fresh interpreter and return {"config": ..., "result": ...}."""
    cmd = [sys.executable, "-m", "tinyllm.bench.worker", json.dumps(asdict(case))]
    # tinygrad prints compile progress to stderr; keep it out of the way unless something fails
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            parsed: dict[str, Any] = json.loads(line[len(RESULT_PREFIX) :])
            return parsed
    tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
    raise RuntimeError(f"benchmark case {case.key()} failed (exit {proc.returncode}):\n{tail}")


def _row(c: dict[str, Any]) -> str:
    cfg, r = c["config"], c["result"]
    p, d = r["prefill"], r["decode"]
    return (
        f"{cfg['rows']:>4} {cfg['chunk']:>5} {cfg['mode']:>7} | "
        f"{p['median_ms']:8.1f} {p['tok_s']:8.1f} {p['pipelined_tok_s']:9.1f} | "
        f"{d['median_ms']:8.2f} {d['tok_s']:8.2f} {d['tok_s_per_row']:8.2f} {d['gb_s']:6.0f} | "
        f"{r['load_s']:6.1f} {r['warmup_s']:7.1f}"
    )


# column widths match _row
HEADER = (
    f"{'rows':>4} {'chunk':>5} {'mode':>7} | {'prefill':>27} | {'decode':>33} | {'start':>14}\n"
    f"{'':18} | {'ms/call':>8} {'tok/s':>8} {'pipelined':>9} | "
    f"{'ms/step':>8} {'tok/s':>8} {'per row':>8} {'GB/s':>6} |"
    f" {'load s':>6} {'warm s':>7}"
)


def main(args: argparse.Namespace) -> int:
    model = find_model(args.model)
    # read the baseline first: --json may point at the same file
    baseline = json.loads(Path(args.compare).read_text()) if args.compare else None
    cases = [
        Case(str(model), args.device, rows, chunk, mode, args.prefill, args.decode, args.ctx)
        for mode in args.mode
        for chunk in args.chunk
        for rows in args.rows
    ]
    print(f"{model.name}: {len(cases)} case(s), each in a fresh process\n{HEADER}", flush=True)
    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(run_case(case))
        print(_row(results[-1]), flush=True)
    doc = record.document(model, results)

    if args.json_out:
        out = Path(args.json_out)
        if args.append and out.exists():
            doc = record.merge(json.loads(out.read_text()), doc)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {out}")

    if baseline is None:
        return 0
    deltas, missing = record.compare(baseline, record.document(model, results))
    print(f"\nagainst {args.compare} (tolerance {args.tolerance:.0%})")
    failed = False
    for d in deltas:
        bad = d.regressed(args.tolerance)
        failed |= bad
        verdict = "REGRESSED" if bad else "ok"
        print(f"  {verdict:9s} {d.case:50s} {d.metric:24s} {d.base:9.2f} -> {d.new:9.2f} ({d.change:+.1%})")
    for key in missing:
        print(f"  no baseline for {key}")
    return 1 if failed else 0
