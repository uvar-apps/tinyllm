"""Oracle records and their comparison. Pure data: no tinygrad, no model.

A record holds, for each prompt, the prompt token ids, the greedy continuation, and for every
generated token the top-k logits plus the log-sum-exp over the full vocabulary (so log
probabilities can be recovered without storing 248k values per step).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = 1
# Same device kind and same pinned tinygrad should reproduce logits almost exactly. Across device
# kinds the AMD decode path quantizes activations to int8, so only loose agreement is expected.
ATOL_SAME_DEVICE = 0.02
ATOL_CROSS_DEVICE = 1.0

DEFAULT_PROMPTS: list[dict[str, str]] = [
    {"id": "capital", "text": "The capital of France is"},
    {"id": "count", "text": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"},
    {"id": "code", "text": "def fibonacci(n):\n    if n < 2:\n        return n\n    return"},
    {"id": "json", "text": '{"name": "Ada Lovelace", "born": 1815, "fields": ["mathematics",'},
    {"id": "multilingual", "text": "Translate to German: The weather is nice today.\nGerman:"},
    {
        "id": "long",
        "text": (
            "The following is a detailed description of how a bill becomes law. "
            "First, a member of the legislature drafts the text and introduces it. "
            "The bill is assigned to a committee, which studies it, holds hearings and may amend it. "
            "If the committee approves, the bill goes to the full chamber for debate and a vote. "
            "After passing one chamber it moves to the other, where the process repeats. "
            "Differences between the two versions are resolved, and both chambers vote again. "
            "Finally the bill is sent to the head of state, who may sign it or veto it. In summary, the"
        ),
    },
]


@dataclass
class Step:
    top_ids: list[int]
    top_logits: list[float]
    logsumexp: float

    def logprob(self, token: int) -> float | None:
        return self.top_logits[self.top_ids.index(token)] - self.logsumexp if token in self.top_ids else None


@dataclass
class PromptRecord:
    id: str
    text: str
    prompt_ids: list[int]
    gen_ids: list[int]
    steps: list[Step]


@dataclass
class OracleRecord:
    model: dict[str, Any]
    device_kind: str  # "amd", "cpu", ...
    tinygrad_rev: str | None
    config: dict[str, int]
    prompts: list[PromptRecord] = field(default_factory=list)
    schema: int = SCHEMA

    def to_json(self) -> str:
        def compact(p: PromptRecord) -> dict[str, Any]:
            return {
                "id": p.id,
                "text": p.text,
                "prompt_ids": p.prompt_ids,
                "gen_ids": p.gen_ids,
                "steps": [
                    {
                        "top_ids": s.top_ids,
                        "top_logits": [round(x, 4) for x in s.top_logits],
                        "logsumexp": round(s.logsumexp, 4),
                    }
                    for s in p.steps
                ],
            }

        doc = {
            "schema": self.schema,
            "model": self.model,
            "device_kind": self.device_kind,
            "tinygrad_rev": self.tinygrad_rev,
            "config": self.config,
            "prompts": [compact(p) for p in self.prompts],
        }
        return json.dumps(doc, indent=1) + "\n"

    @staticmethod
    def from_json(text: str) -> OracleRecord:
        d = json.loads(text)
        if d.get("schema") != SCHEMA:
            raise ValueError(f"unsupported oracle record schema {d.get('schema')}")
        prompts = [
            PromptRecord(p["id"], p["text"], p["prompt_ids"], p["gen_ids"], [Step(**s) for s in p["steps"]])
            for p in d["prompts"]
        ]
        return OracleRecord(d["model"], d["device_kind"], d["tinygrad_rev"], d["config"], prompts)

    @staticmethod
    def load(path: str | Path) -> OracleRecord:
        return OracleRecord.from_json(Path(path).read_text())


@dataclass
class Comparison:
    prompts: int = 0
    steps: int = 0
    tokenizer_mismatches: list[str] = field(default_factory=list)
    token_mismatches: list[str] = field(default_factory=list)  # "prompt@step: expected -> got"
    max_logit_diff: float = 0.0
    logit_violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.tokenizer_mismatches or self.token_mismatches or self.logit_violations)

    def summary(self) -> str:
        lines = [
            f"{self.prompts} prompts, {self.steps} steps compared, largest logit difference {self.max_logit_diff:.4f}"
        ]
        for label, items in (
            ("tokenizer differs", self.tokenizer_mismatches),
            ("greedy token differs", self.token_mismatches),
            ("logits out of tolerance", self.logit_violations),
        ):
            lines += [f"  {label}: {item}" for item in items[:10]]
            if len(items) > 10:
                lines.append(f"  {label}: ... and {len(items) - 10} more")
        return "\n".join(lines)


def compare(golden: OracleRecord, fresh: OracleRecord, atol: float | None = None, top: int = 4) -> Comparison:
    """Compare a fresh capture with a stored one.

    Tokens must match exactly up to the first divergence of each prompt (after which the two runs
    are on different paths and are not compared). Logits are compared for the golden run's `top`
    highest-ranked tokens at each step.
    """
    if atol is None:
        atol = ATOL_SAME_DEVICE if golden.device_kind == fresh.device_kind else ATOL_CROSS_DEVICE
    result = Comparison()
    fresh_by_id = {p.id: p for p in fresh.prompts}
    for g in golden.prompts:
        f = fresh_by_id.get(g.id)
        if f is None:
            result.tokenizer_mismatches.append(f"{g.id}: missing from the fresh capture")
            continue
        result.prompts += 1
        if g.prompt_ids != f.prompt_ids:
            result.tokenizer_mismatches.append(f"{g.id}: {len(g.prompt_ids)} vs {len(f.prompt_ids)} prompt tokens")
            continue
        for i, (gs, fs) in enumerate(zip(g.steps, f.steps, strict=False)):
            result.steps += 1
            for token, logit in list(zip(gs.top_ids, gs.top_logits, strict=True))[:top]:
                if token in fs.top_ids:
                    diff = abs(fs.top_logits[fs.top_ids.index(token)] - logit)
                    result.max_logit_diff = max(result.max_logit_diff, diff)
                    if diff > atol:
                        result.logit_violations.append(f"{g.id}@{i}: token {token} differs by {diff:.4f}")
                elif token == gs.top_ids[0]:
                    result.logit_violations.append(f"{g.id}@{i}: expected top token {token} is not in the fresh top-k")
            if g.gen_ids[i] != f.gen_ids[i]:
                result.token_mismatches.append(f"{g.id}@{i}: {g.gen_ids[i]} -> {f.gen_ids[i]}")
                break
    return result
