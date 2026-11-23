"""From generated tokens to OpenAI-style deltas, for one request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tinyllm.engine.sampling import Sample
from tinyllm.frontend.detokenizer import Detokenizer
from tinyllm.frontend.router import StreamRouter, ToolCall
from tinyllm.frontend.tokenizer import Tokenizer


@dataclass
class Delta:
    """What one step adds to the response."""

    content: str = ""
    reasoning: str = ""
    logprobs: list[dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.content or self.reasoning or self.logprobs)


class OutputProcessor:
    """Detokenizes, applies stop strings, and (for chat) routes reasoning, content and tool calls.

    `stopped` turns true once a stop string matched; the caller then cancels the request.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        stop: Sequence[str] = (),
        chat: bool = True,
        reasoning: bool = False,
        tools: Sequence[Mapping[str, Any]] | None = None,
        logprobs: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.detokenizer = Detokenizer(tokenizer.token_bytes, stop)
        self.router = StreamRouter(reasoning=reasoning, tools=tools) if chat else None
        self.want_logprobs = logprobs
        self.text = ""  # all content emitted
        self.reasoning = ""

    @property
    def stopped(self) -> bool:
        return self.detokenizer.stopped is not None

    def token(self, token: int, sample: Sample) -> Delta:
        delta = self._route(self.detokenizer.push(token), final=False)
        if self.want_logprobs and sample.logprob is not None:
            delta.logprobs.append(self._entry(token, sample.logprob, sample.top))
        return delta

    def finish(self) -> tuple[Delta, list[ToolCall]]:
        """Text held back until the end, and the tool calls, if any."""
        delta = self._route(self.detokenizer.finish(), final=True)
        if self.router is None:
            return delta, []
        calls, unparsed = self.router.finish()
        if unparsed:  # never silently drop output the client cannot use as a call
            delta.content += unparsed
            self.text += unparsed
        return delta, calls

    def _route(self, text: str, final: bool) -> Delta:
        delta = Delta()
        if self.router is None:
            delta.content = text
        else:
            for kind, piece in self.router.route(text, final=final):
                if kind == "reasoning_content":
                    delta.reasoning += piece
                else:
                    delta.content += piece
        self.text += delta.content
        self.reasoning += delta.reasoning
        return delta

    def _entry(self, token: int, logprob: float, top: Sequence[tuple[int, float]]) -> dict[str, Any]:
        def one(t: int, lp: float) -> dict[str, Any]:
            raw = self.tokenizer.token_bytes(t)
            return {"token": raw.decode(errors="replace"), "logprob": lp, "bytes": list(raw)}

        return {**one(token, logprob), "top_logprobs": [one(t, lp) for t, lp in top]}
