"""`tinyllm oracle capture | check | reference`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.oracle.record import DEFAULT_PROMPTS, OracleRecord, compare
from tinyllm.weights.resolve import find_model

# share of positions where llama.cpp must pick the same most-likely token
REFERENCE_TOP1_MIN = 0.9


def _capture(args: argparse.Namespace) -> int:
    from tinyllm.oracle.capture import capture

    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else DEFAULT_PROMPTS
    record = capture(find_model(args.model), prompts, args.decode, args.ctx, topk=args.topk, device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(record.to_json())
    steps = sum(len(p.steps) for p in record.prompts)
    print(f"wrote {out}: {len(record.prompts)} prompts, {steps} steps, device {record.device_kind}")
    return 0


def _check(args: argparse.Namespace) -> int:
    from tinyllm.oracle.capture import capture

    golden = OracleRecord.load(args.golden)
    model = find_model(args.model or golden.model["file"])
    prompts = [{"id": p.id, "text": p.text} for p in golden.prompts]
    cfg = golden.config
    fresh = capture(model, prompts, cfg["decode"], cfg["max_context"], cfg["chunk"], cfg["topk"], args.device)
    if fresh.model["fingerprint"] != golden.model["fingerprint"]:
        print(f"warning: {model.name} is not the file the record was captured from")
    if fresh.tinygrad_rev != golden.tinygrad_rev:
        print(
            f"note: record captured on tinygrad {str(golden.tinygrad_rev)[:12]}, running {str(fresh.tinygrad_rev)[:12]}"
        )
    result = compare(golden, fresh, args.atol)
    kind = (
        "same device kind"
        if golden.device_kind == fresh.device_kind
        else f"{golden.device_kind} record vs {fresh.device_kind} run"
    )
    print(f"{Path(args.golden).name} ({kind}): {'OK' if result.ok else 'MISMATCH'}")
    print(result.summary())
    return 0 if result.ok else 1


def _reference(args: argparse.Namespace) -> int:
    from tinyllm.oracle import llamacpp

    golden = OracleRecord.load(args.golden)
    model = find_model(args.model or golden.model["file"])
    binary = Path(args.server).expanduser() if args.server else None
    with llamacpp.ReferenceServer(model, binary, golden.config["max_context"], args.threads) as server:
        report = llamacpp.compare(golden, server, golden.config["topk"])
    print(f"{Path(args.golden).name} ({golden.device_kind} record)")
    print(report.summary())
    ok = report.top1_rate >= REFERENCE_TOP1_MIN and not report.tokenizer_mismatches
    return 0 if ok else 1


def main(args: argparse.Namespace) -> int:
    return {"capture": _capture, "check": _check, "reference": _reference}[args.oracle_command](args)
