"""Chat templates: the model's own Jinja template, rendered in Jinja's immutable sandbox.

Model files are untrusted input (design 9), so the template cannot reach Python internals or mutate the request.
The environment matches what Hugging Face's `apply_chat_template` provides, which is what the templates in GGUF
files are written for: `raise_exception`, `strftime_now`, `bos_token`, `eos_token`, a `tojson` that does not escape
HTML characters, and the loop-control extension.
"""

from __future__ import annotations

import contextlib
import copy
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

__all__ = ["ChatTemplate", "TemplateError", "TemplateRenderError", "normalize_messages"]


class TemplateRenderError(ValueError):
    """The template rejected the conversation (it called raise_exception, or failed)."""


def _raise(message: str) -> None:
    raise TemplateRenderError(message)


def _tojson(value: Any, indent: int | None = None, **_: Any) -> str:
    return json.dumps(value, indent=indent, ensure_ascii=False)


def normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A copy of OpenAI-style messages in the form chat templates expect.

    Tool-call arguments arrive as JSON strings and templates need them as objects; `content: null` (an assistant
    turn that only calls tools) becomes "".
    """
    out = [dict(copy.deepcopy(m)) for m in messages]
    for m in out:
        for call in m.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(arguments := function.get("arguments"), str):
                with contextlib.suppress(json.JSONDecodeError):  # else left as a string for the template to judge
                    function["arguments"] = json.loads(arguments) if arguments.strip() else {}
        if m.get("content", "") is None:
            m["content"] = ""
    return out


class ChatTemplate:
    def __init__(self, source: str, bos_token: str = "", eos_token: str = "") -> None:
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = _tojson
        env.globals.update(
            raise_exception=_raise,
            strftime_now=lambda fmt: time.strftime(fmt),
            bos_token=bos_token,
            eos_token=eos_token,
        )
        self.source = source
        self._template = env.from_string(source)

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        """The prompt text. `kwargs` are the request's `chat_template_kwargs`, such as `enable_thinking`."""
        try:
            return self._template.render(
                messages=normalize_messages(messages),
                tools=list(tools) if tools else None,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except TemplateRenderError:
            raise
        except TemplateError as e:
            raise TemplateRenderError(f"chat template failed: {e}") from e
