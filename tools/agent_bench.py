#!/usr/bin/env python3
"""Time a coding agent's conversation against a running server: how long each step waits for its first token.

    uv run python tools/agent_bench.py --base-url http://127.0.0.1:8000/v1 [--steps 8] [--echo-reasoning]

A coding agent sends the whole conversation on every step: a long system prompt with tool definitions, the task,
then for each step the model's reply and the tool's output (here, Python and Rust files of a made-up repository).
The server can only skip the part of a prompt its state already holds. Clients that do not send the model's
reasoning back (the default here; `--echo-reasoning` sends it) get each reply re-rendered differently from how it
came out, which costs a full prefill on every step unless the server can resume from a checkpoint
(tinyllm.runner.checkpoints).

Greedy decoding with thinking on, as agents run it. Prints, per step, the prompt and cached tokens, the time to
first token and the decode rate, then the totals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file of the repository and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the repository root"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Replace a file of the repository with new contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite (cargo test or pytest) and return the output.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "A test name filter, or empty for all"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the repository for a regular expression; returns matching lines with paths.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]

RULES = """You are a coding agent working in a repository that mixes a Rust core library (crates/) and Python
bindings and tools (python/). You can read files, write files, search and run the tests through the tools.

Rules:
- Read the relevant code before changing it. Never guess at the contents of a file.
- Keep changes minimal and in the style of the surrounding code. Do not reformat unrelated lines.
- Rust: no unsafe code, no unwrap() outside tests, errors through the crate's Error type, clippy clean.
- Python: type hints everywhere, no bare except, tests with pytest, ruff clean.
- After every change run the tests that cover it, and report what you ran and the result.
- When a test fails, read the failure carefully, form a hypothesis, and check it against the code before editing.
- Explain each change in one or two sentences when you are done.
"""


def rust_file(name: str, n: int) -> str:
    """A plausible Rust module of about a thousand tokens."""
    fns = []
    for i in range(n):
        fns.append(f"""
    /// Parses the {i}th field of a {name} record, rejecting values outside the allowed range.
    pub fn parse_field_{i}(&self, input: &str) -> Result<u32, Error> {{
        let trimmed = input.trim();
        if trimmed.is_empty() {{
            return Err(Error::Empty {{ field: "{name}_{i}" }});
        }}
        let value: u32 = trimmed.parse().map_err(|_| Error::NotANumber(trimmed.to_string()))?;
        if value > self.limits[{i % 4}] {{
            return Err(Error::OutOfRange {{ value, max: self.limits[{i % 4}] }});
        }}
        Ok(value * {i + 1})
    }}
""")
    tests = "\n".join(
        f"""    #[test]
    fn parses_field_{i}() {{
        let p = {name.capitalize()}Parser::new([10, 20, 30, 40]);
        assert_eq!(p.parse_field_{i}(" 3 ").unwrap(), {3 * (i + 1)});
        assert!(p.parse_field_{i}("").is_err());
    }}"""
        for i in range(n)
    )
    return f"""use crate::error::Error;

/// Parser for {name} records: one field per line, each checked against a per-column limit.
#[derive(Debug, Clone)]
pub struct {name.capitalize()}Parser {{
    limits: [u32; 4],
}}

impl {name.capitalize()}Parser {{
    pub fn new(limits: [u32; 4]) -> Self {{
        Self {{ limits }}
    }}
{"".join(fns)}}}

#[cfg(test)]
mod tests {{
    use super::*;

{tests}
}}
"""


def python_file(name: str, n: int) -> str:
    """A plausible Python module of about a thousand tokens."""
    methods = "".join(
        f'''
    def load_{i}(self, rows: list[dict[str, str]]) -> list[int]:
        """Convert column {i} of the {name} rows, skipping blanks and rejecting negative values."""
        out: list[int] = []
        for row in rows:
            raw = row.get("col_{i}", "").strip()
            if not raw:
                continue
            value = int(raw)
            if value < 0:
                raise ValueError(f"negative value {{value}} in col_{i}")
            out.append(value * self.scale)
        return out
'''
        for i in range(n)
    )
    tests = "\n".join(
        f"""
def test_load_{i}() -> None:
    loader = {name.capitalize()}Loader(scale=2)
    assert loader.load_{i}([{{"col_{i}": " 4 "}}, {{"col_{i}": ""}}]) == [8]
"""
        for i in range(n)
    )
    return f'''"""Loaders for {name} exports: typed columns out of the CSV rows the Rust core produces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class {name.capitalize()}Loader:
    scale: int = 1
{methods}
{tests}'''


FILES = {
    "crates/core/src/orders.rs": rust_file("orders", 6),
    "python/tools/orders.py": python_file("orders", 7),
    "crates/core/src/invoices.rs": rust_file("invoices", 6),
    "python/tools/invoices.py": python_file("invoices", 7),
    "crates/core/src/shipping.rs": rust_file("shipping", 6),
    "python/tools/shipping.py": python_file("shipping", 7),
    "crates/core/src/returns.rs": rust_file("returns", 6),
    "python/tools/returns.py": python_file("returns", 7),
}
TASK = (
    "The nightly job reports that parse_field_3 in crates/core/src/orders.rs accepts values above the column "
    "limit, and the Python loader in python/tools/orders.py silently drops rows it should reject. Find the bug in "
    "both, fix it, and check the other parsers and loaders for the same mistake."
)


def system_prompt() -> str:
    """About 4,000 tokens, as agents' system prompts run (theirs are often longer): rules, the layout, the error
    type and one file of each language as an example of the house style."""
    tree = "\n".join(f"  {p}  ({len(c.splitlines())} lines)" for p, c in FILES.items())
    rust, python = rust_file("ledger", 8), python_file("ledger", 8)
    style = f"crates/core/src/ledger.rs:\n\n{rust}\npython/tools/ledger.py:\n\n{python}"
    return (
        f"{RULES}\nRepository layout:\n{tree}\n\nThe Rust error type, for reference:\n\n{FILES_ERROR}\n"
        f"Examples of the house style:\n\n{style}"
    )


FILES_ERROR = """#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("field {field} is empty")]
    Empty { field: &'static str },
    #[error("{0:?} is not a number")]
    NotANumber(String),
    #[error("{value} is above the limit {max}")]
    OutOfRange { value: u32, max: u32 },
}
"""


def tool_result(call: dict[str, Any], order: list[str]) -> str:
    """What the tool returns: the file asked for, or the next unread one."""
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}
    name = call["function"]["name"]
    if name == "read_file" and args.get("path") in FILES:
        return FILES[args["path"]]
    if name == "run_tests":
        return "running 24 tests\ntest orders::tests::parses_field_3 ... FAILED\n\n" + FILES[order.pop(0)]
    if name == "write_file":
        return "ok"
    return FILES[order.pop(0)] if order else "no matches"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400, help="tokens per reply")
    ap.add_argument("--echo-reasoning", action="store_true", help="send each reply's reasoning_content back")
    ap.add_argument("--json", default=None, help="write the per-step numbers here")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=900)
    model = client.models.list().data[0].id
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": TASK},
    ]
    order = list(FILES)
    rows = []
    print(f"{model}: reasoning {'sent back' if args.echo_reasoning else 'not sent back'}")
    print(" step  prompt  cached  prefilled   ttft s  generated  decode tok/s  reply")
    for step in range(args.steps):
        start = time.perf_counter()
        first = None
        content, reasoning, calls, usage = [], [], {}, None
        stream = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=args.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "reasoning_content", None) or ""
            if first is None and (delta.content or text or delta.tool_calls):
                first = time.perf_counter()
            reasoning.append(text)
            content.append(delta.content or "")
            for tc in delta.tool_calls or []:
                entry = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                entry["id"] = tc.id or entry["id"]
                if tc.function is not None:
                    entry["name"] += tc.function.name or ""
                    entry["arguments"] += tc.function.arguments or ""
        end = time.perf_counter()
        assert usage is not None, "the server sent no usage"
        cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
        ttft = (first or end) - start
        decode = (usage.completion_tokens - 1) / (end - first) if first and usage.completion_tokens > 1 else 0.0
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if args.echo_reasoning and "".join(reasoning):
            message["reasoning_content"] = "".join(reasoning)
        tool_calls = [
            {
                "id": c["id"] or f"call_{step}_{i}",
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for i, c in sorted(calls.items())
        ]
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)
        if tool_calls:
            for c in tool_calls:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": tool_result(c, order)})
            what = ", ".join(f"{c['function']['name']}({c['function']['arguments'][:40]})" for c in tool_calls)
        else:
            path = order.pop(0) if order else "crates/core/src/orders.rs"
            messages.append({"role": "user", "content": f"Here is {path}; check it for the same bug.\n\n{FILES[path]}"})
            what = repr("".join(content)[:50])
        row = {
            "step": step,
            "prompt": usage.prompt_tokens,
            "cached": cached,
            "ttft": ttft,
            "generated": usage.completion_tokens,
            "decode": decode,
        }
        rows.append(row)
        print(
            f"{step:5d}  {usage.prompt_tokens:6d}  {cached:6d}  {usage.prompt_tokens - cached:9d}  {ttft:7.2f}  "
            f"{usage.completion_tokens:9d}  {decode:12.1f}  {what}"
        )
    prompt = sum(r["prompt"] for r in rows)
    cached = sum(r["cached"] for r in rows)
    ttft = sum(r["ttft"] for r in rows)
    print(
        f"total  {prompt:6d}  {cached:6d}  {prompt - cached:9d}  {ttft:7.2f}  (steps after the first: "
        f"{sum(r['ttft'] for r in rows[1:]):.2f} s waiting for the first token)"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"echo_reasoning": args.echo_reasoning, "steps": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
