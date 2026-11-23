"""Routing generated text into reasoning, content and tool calls.

Adapted from tinygrad's `tinygrad/llm/serve.py` (`StreamRouter`, `parse_tool_call`; MIT licence) and narrowed to
the format Qwen3.x uses: reasoning between `<think>` and `</think>`, tool calls after the answer as
`<tool_call>` blocks holding either `<function=name><parameter=key>value</parameter>...</function>` (Qwen3.5 and
later) or a JSON object `{"name": ..., "arguments": ...}` (Qwen3 and Hermes-style templates).

Text is routed as it streams, holding back a partial tag. Tool calls are parsed once generation ends, and their
parameter values are converted with the request's tool schemas: a parameter declared as a string stays a string
even if it looks like a number.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

THINK, END_THINK, TOOL_CALL = "<think>", "</think>", "<tool_call>"
CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
XML_CALL = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*(?:</function>)?\s*$", re.DOTALL)
XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as OpenAI's API returns it


def _schema_types(tools: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, str]]:
    """function name -> parameter name -> JSON-schema type, from OpenAI-style tool definitions."""
    types: dict[str, dict[str, str]] = {}
    for tool in tools or ():
        function = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        properties = (
            ((function.get("parameters") or {}).get("properties") or {}) if isinstance(function, Mapping) else {}
        )
        types[str(function.get("name"))] = {
            str(k): str(v.get("type", "")) if isinstance(v, Mapping) else "" for k, v in properties.items()
        }
    return types


def _convert(value: str, kind: str) -> Any:
    if kind == "string":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_tool_call(text: str, types: Mapping[str, Mapping[str, str]] | None = None) -> tuple[str, Any] | None:
    """(function name, arguments) from the inside of one `<tool_call>` block, or None if it is neither format."""
    text = text.strip()
    if text.startswith("{"):
        try:
            call = json.loads(text)
            return str(call["name"]), call.get("arguments", call.get("parameters", {}))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    if (m := XML_CALL.match(text)) is None:
        return None
    name, body = m.group(1), m.group(2)
    kinds = (types or {}).get(name, {})
    arguments = {}
    for p in XML_PARAMETER.finditer(body):
        value = re.sub(r"^\r?\n|\r?\n\Z", "", p.group(2))  # the format puts each value on its own lines
        arguments[p.group(1)] = _convert(value, kinds.get(p.group(1), ""))
    if XML_PARAMETER.sub("", body).strip():  # text outside any parameter: not a well-formed call
        return None
    return name, arguments


class StreamRouter:
    """Splits streamed text into ("reasoning_content" | "content", text) deltas; tool-call text is kept apart."""

    def __init__(self, reasoning: bool = False, tools: Sequence[Mapping[str, Any]] | None = None) -> None:
        # `reasoning`: the prompt already opened a think block, so output starts inside it
        self.mode = "reasoning" if reasoning else "undecided"
        self.tools = list(tools) if tools else []
        self.buf = ""
        self.tool_text = ""  # everything from the first <tool_call> on
        self._after_think = False  # drop the newlines the template puts between </think> and the answer

    def _split(self, tag: str, final: bool) -> tuple[str, bool]:
        """Text before the first `tag` in the buffer, and whether it was found; holds back a partial tag."""
        if tag in self.buf:
            before, self.buf = self.buf.split(tag, 1)
            return before, True
        hold = 0
        if not final:
            hold = max(
                (i for i in range(1, min(len(self.buf), len(tag)) + 1) if tag.startswith(self.buf[-i:])), default=0
            )
        emit, self.buf = self.buf[: len(self.buf) - hold], self.buf[len(self.buf) - hold :]
        return emit, False

    def route(self, piece: str, final: bool = False) -> Iterator[tuple[str, str]]:
        self.buf += piece
        if self.mode == "undecided":  # does the output open with a think block?
            if not final and len(self.buf) < len(THINK) and THINK.startswith(self.buf):
                return
            if self.buf.startswith(THINK):
                self.mode, self.buf = "reasoning", self.buf[len(THINK) :]
            else:
                self.mode = "content"
        if self.mode == "reasoning":
            emit, done = self._split(END_THINK, final)
            if emit:
                yield "reasoning_content", emit
            if not done:
                return
            self.mode, self._after_think = "content", True
        if self._after_think:
            self.buf = self.buf.lstrip("\n")
            if not self.buf and not final:
                return
            self._after_think = False
        if self.mode == "tool":
            self.tool_text += self.buf
            self.buf = ""
            return
        if not self.tools:  # without tools in the request, a <tool_call> is just text
            if self.buf:
                yield "content", self.buf
            self.buf = ""
            return
        emit, found = self._split(TOOL_CALL, final)
        if emit:
            yield "content", emit
        if found:
            self.mode, self.tool_text, self.buf = "tool", TOOL_CALL + self.buf, ""

    def finish(self) -> tuple[list[ToolCall], str]:
        """The tool calls in the output, and any tool-call text that did not parse (to return as content)."""
        types = _schema_types(self.tools)
        calls, unparsed = [], []
        for m in CALL_BLOCK.finditer(self.tool_text):
            parsed = parse_tool_call(m.group(1), types)
            if parsed is None:
                unparsed.append(m.group(0))
                continue
            name, arguments = parsed
            text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:24]}", name, text))
        return calls, "".join(unparsed)
