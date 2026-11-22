"""Capture an oracle record by running tinygrad's own engine greedily in logits mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tinyllm.oracle.record import OracleRecord, PromptRecord, Step
from tinyllm.weights.resolve import fingerprint


def _step(logits: Any, topk: int) -> Step:
    import numpy as np

    row = np.asarray(logits, dtype=np.float64).reshape(-1)
    order = np.argsort(-row, kind="stable")[:topk]
    peak = float(row.max())
    return Step(
        top_ids=[int(i) for i in order],
        top_logits=[float(row[i]) for i in order],
        logsumexp=peak + float(np.log(np.exp(row - peak).sum())),
    )


def capture_session(session: Any, model: Path, prompts: list[dict[str, str]], decode: int, topk: int) -> OracleRecord:
    """Capture with an existing single-row logits-mode `UpstreamSession`."""
    from tinyllm import tg

    if session.rows != 1 or not session.logits:
        raise ValueError("oracle capture needs a single-row session in logits mode")
    tokenizer = session.tokenizer()
    record = OracleRecord(
        model={"file": model.name, "bytes": model.stat().st_size, "fingerprint": fingerprint(model)},
        device_kind=tg.default_device().split(":")[0].lower(),
        tinygrad_rev=tg.TINYGRAD_REV,
        config={"decode": decode, "max_context": session.max_context, "chunk": session.chunk, "topk": topk},
    )
    for prompt in prompts:
        ids = [int(t) for t in tokenizer.encode(prompt["text"])]
        session.reset()
        logits = session.prefill(ids)
        steps: list[Step] = []
        generated: list[int] = []
        for i in range(decode):
            step = _step(logits[0], topk)
            steps.append(step)
            generated.append(step.top_ids[0])
            if i + 1 < decode:
                logits = session.step([step.top_ids[0]])
        record.prompts.append(PromptRecord(prompt["id"], prompt["text"], ids, generated, steps))
    return record


def capture(
    model: Path,
    prompts: list[dict[str, str]],
    decode: int = 16,
    max_context: int = 1024,
    chunk: int = 32,
    topk: int = 8,
    device: str | None = None,
) -> OracleRecord:
    from tinyllm import device as device_mod

    device_mod.configure(device)
    from tinyllm.tg.upstream import UpstreamSession

    session = UpstreamSession(model, max_context, rows=1, chunk=chunk, logits=True)
    return capture_session(session, model, prompts, decode, topk)
