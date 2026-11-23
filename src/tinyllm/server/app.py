"""OpenAI-compatible routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`.

Requests are validated loosely (unknown fields are ignored, as OpenAI-compatible servers usually do), rendered
and tokenized here, and run by the engine through `EngineClient`. Responses stream as server-sent events when
`stream` is set. A client that disconnects cancels its request; a full queue answers 429.

Sampling defaults come from the model file (`general.sampling.*`) for whatever the request leaves unset.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from tinyllm.engine.core import EngineRequest, FinishEvent, TokenEvent
from tinyllm.engine.sampling import SamplingParams
from tinyllm.frontend.output import Delta, OutputProcessor
from tinyllm.frontend.router import ToolCall
from tinyllm.frontend.template import ChatTemplate, TemplateRenderError
from tinyllm.frontend.tokenizer import Tokenizer
from tinyllm.server.client import EngineClient, QueueFull

DISCONNECT_POLL_SECONDS = 0.5


@dataclass
class ServerConfig:
    model_name: str
    max_context: int
    sampling_defaults: dict[str, float] = field(default_factory=dict)  # temperature, top_k, top_p
    api_key: str | None = None
    queue_timeout: float | None = None


class APIError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error", code: str | None = None,
                 param: str | None = None) -> None:  # fmt: skip
        super().__init__(message)
        self.status, self.message, self.kind, self.code, self.param = status, message, kind, code, param

    def response(self) -> JSONResponse:
        body = {"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}}
        return JSONResponse(body, status_code=self.status)


class _Common(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None  # not in OpenAI's API; vLLM and llama.cpp accept it
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    n: int = 1


class ChatRequest(_Common):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    max_completion_tokens: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class CompletionRequest(_Common):
    prompt: str | list[int]
    logprobs: int | None = None  # here the number of top tokens to report, as in OpenAI's legacy API
    echo: bool = False


@dataclass
class _Outcome:
    reason: str
    usage: dict[str, Any]
    calls: list[ToolCall]
    error: str | None = None


def create_app(
    client: EngineClient, tokenizer: Tokenizer, template: ChatTemplate | None, config: ServerConfig
) -> FastAPI:
    app = FastAPI(title="tinyllm", docs_url=None, redoc_url=None)

    @app.exception_handler(APIError)
    async def _api_error(_: Request, e: APIError) -> JSONResponse:
        return e.response()

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError) -> JSONResponse:
        return APIError(400, f"invalid request: {e.errors()}").response()

    async def authorize(request: Request) -> None:
        if config.api_key and request.headers.get("authorization", "") != f"Bearer {config.api_key}":
            raise APIError(401, "invalid or missing API key", "authentication_error", "invalid_api_key")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        running, waiting = client.engine.load
        return {"status": "ok", "running": running, "waiting": waiting}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(client.engine.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict[str, Any]:
        entry = {"id": config.model_name, "object": "model", "created": 0, "owned_by": "tinyllm"}
        return {"object": "list", "data": [{**entry, "max_model_len": config.max_context}]}

    def sampling(body: _Common, logprobs: bool, top_logprobs: int) -> SamplingParams:
        d = config.sampling_defaults

        def pick(value: Any, key: str, default: Any) -> Any:
            return value if value is not None else d.get(key, default)

        try:
            return SamplingParams(
                temperature=float(pick(body.temperature, "temperature", 1.0)),
                top_k=int(pick(body.top_k, "top_k", 0)),
                top_p=float(pick(body.top_p, "top_p", 1.0)),
                min_p=float(body.min_p or 0.0),
                repetition_penalty=float(body.repetition_penalty or 1.0),
                presence_penalty=float(body.presence_penalty or 0.0),
                frequency_penalty=float(body.frequency_penalty or 0.0),
                logit_bias={int(k): float(v) for k, v in (body.logit_bias or {}).items()},
                seed=body.seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )
        except ValueError as e:
            raise APIError(400, str(e)) from e

    def limits(body: _Common, prompt: list[int], max_tokens: int | None) -> int:
        if body.n != 1:
            raise APIError(400, "only n=1 is supported", param="n")
        if len(prompt) >= config.max_context:
            message = f"prompt has {len(prompt)} tokens, but the model context is {config.max_context}"
            raise APIError(400, message, code="context_length_exceeded", param="messages")
        room = config.max_context - len(prompt) - 1
        if max_tokens is not None and max_tokens < 1:
            raise APIError(400, "max_tokens must be at least 1", param="max_tokens")
        return min(max_tokens, room) if max_tokens is not None else room

    def stops(body: _Common) -> list[str]:
        return [body.stop] if isinstance(body.stop, str) else list(body.stop or [])

    async def run(request: EngineRequest, out: OutputProcessor) -> AsyncIterator[Delta | _Outcome]:
        completion, finish = 0, None
        try:
            async for event in client.generate(request):
                if isinstance(event, TokenEvent):
                    completion += 1
                    delta = out.token(event.token, event.sample)
                    if delta:
                        yield delta
                    if out.stopped:  # a stop string matched: the rest is not wanted
                        client.cancel(request.id)
                        break
                else:
                    finish = event
        except QueueFull:
            raise APIError(429, "the server is busy; try again later", "rate_limit_error", "queue_full") from None
        tail, calls = out.finish()
        if tail:
            yield tail
        reason = "stop" if out.stopped or finish is None else finish.reason
        error = finish.error if isinstance(finish, FinishEvent) else None
        if reason == "stop" and calls:
            reason = "tool_calls"
        usage = {
            "prompt_tokens": len(request.prompt),
            "completion_tokens": finish.completion_tokens if finish is not None else completion,
            "total_tokens": len(request.prompt) + (finish.completion_tokens if finish is not None else completion),
            "prompt_tokens_details": {"cached_tokens": finish.cached_tokens if finish is not None else 0},
        }
        yield _Outcome(reason, usage, calls, error)

    async def until_disconnect(request: Request, work: Awaitable[Any]) -> Any:
        """Await `work`, cancelling it if the client goes away (non-streaming responses)."""
        task = asyncio.ensure_future(work)
        while True:
            done, _ = await asyncio.wait({task}, timeout=DISCONNECT_POLL_SECONDS)
            if done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                raise APIError(499, "client disconnected")

    def sse(chunks: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
        async def body() -> AsyncIterator[str]:
            try:
                async for chunk in chunks:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except APIError as e:  # after the response started an error can only be another event
                yield f"data: {json.dumps({'error': {'message': e.message, 'type': e.kind, 'code': e.code}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    async def first_or_error(stream: AsyncIterator[Any]) -> tuple[Any, AsyncIterator[Any]]:
        """Run a stream to its first item so that errors before any output (429, 400) become HTTP statuses."""
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            first = None

        async def rest() -> AsyncIterator[Any]:
            if first is not None:
                yield first
            async for item in stream:
                yield item

        return first, rest()

    # ---- chat ------------------------------------------------------------------------------------------------

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat(body: ChatRequest, http: Request) -> Response:
        if template is None:
            raise APIError(400, "this model has no chat template; use /v1/completions")
        kwargs = dict(body.chat_template_kwargs or {})
        if body.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", body.reasoning_effort)
        tools = body.tools if body.tool_choice != "none" else None
        try:
            prompt_text = template.render(body.messages, tools=tools, add_generation_prompt=True, **kwargs)
        except TemplateRenderError as e:
            raise APIError(400, str(e), param="messages") from e
        prompt = tokenizer.encode(prompt_text)
        max_tokens = limits(body, prompt, body.max_completion_tokens or body.max_tokens)
        top = int(body.top_logprobs or 0)
        params = sampling(body, logprobs=body.logprobs, top_logprobs=top if body.logprobs else 0)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        engine_request = EngineRequest(
            request_id, prompt, params, max_tokens, tokenizer.end_tokens, queue_timeout=config.queue_timeout
        )
        reasoning = prompt_text.rstrip().endswith("<think>")  # the prompt opened a think block
        out = OutputProcessor(
            tokenizer, stops(body), chat=True, reasoning=reasoning, tools=tools, logprobs=body.logprobs
        )
        base = {"id": request_id, "created": int(time.time()), "model": body.model or config.model_name}
        events = run(engine_request, out)
        if body.stream:
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            _, events = await first_or_error(events)  # a full queue or a bad request is an HTTP status, not an event
            return sse(chat_chunks(events, base, include_usage))
        return JSONResponse(await until_disconnect(http, chat_whole(events, base)))

    async def chat_chunks(
        events: AsyncIterator[Delta | _Outcome], base: dict[str, Any], include_usage: bool
    ) -> AsyncIterator[dict[str, Any]]:
        chunk = {**base, "object": "chat.completion.chunk"}

        def choice(delta: dict[str, Any], finish: str | None = None, logprobs: Any = None) -> dict[str, Any]:
            return {**chunk, "choices": [{"index": 0, "delta": delta, "logprobs": logprobs, "finish_reason": finish}]}

        yield choice({"role": "assistant", "content": ""})
        async for item in events:
            if isinstance(item, Delta):
                delta: dict[str, Any] = {}
                if item.reasoning:
                    delta["reasoning_content"] = item.reasoning
                if item.content:
                    delta["content"] = item.content
                yield choice(delta, logprobs={"content": item.logprobs} if item.logprobs else None)
            else:
                if item.error:
                    raise APIError(500, item.error, "server_error", "generation_failed")
                if item.calls:
                    yield choice({"tool_calls": [{"index": i, **_call(c)} for i, c in enumerate(item.calls)]})
                yield choice({}, item.reason)
                if include_usage:
                    yield {**chunk, "choices": [], "usage": item.usage}

    async def chat_whole(events: AsyncIterator[Delta | _Outcome], base: dict[str, Any]) -> dict[str, Any]:
        content, reasoning, logprobs = [], [], []
        async for item in events:
            if isinstance(item, Delta):
                content.append(item.content)
                reasoning.append(item.reasoning)
                logprobs += item.logprobs
                continue
            if item.error:
                raise APIError(500, item.error, "server_error", "generation_failed")
            message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
            if "".join(reasoning):
                message["reasoning_content"] = "".join(reasoning)
            if item.calls:
                message["tool_calls"] = [_call(c) for c in item.calls]
            choice = {"index": 0, "message": message, "logprobs": {"content": logprobs} if logprobs else None,
                      "finish_reason": item.reason}  # fmt: skip
            return {**base, "object": "chat.completion", "choices": [choice], "usage": item.usage}
        raise APIError(500, "generation ended without a result", "server_error")

    # ---- text completions -------------------------------------------------------------------------------------

    @app.post("/v1/completions", dependencies=[Depends(authorize)])
    async def completions(body: CompletionRequest, http: Request) -> Response:
        if body.echo:
            raise APIError(400, "echo is not supported", param="echo")
        prompt = tokenizer.encode(body.prompt) if isinstance(body.prompt, str) else [int(t) for t in body.prompt]
        if not prompt:
            raise APIError(400, "the prompt is empty", param="prompt")
        max_tokens = limits(body, prompt, body.max_tokens if body.max_tokens is not None else 16)
        top = int(body.logprobs or 0)
        params = sampling(body, logprobs=body.logprobs is not None, top_logprobs=top)
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        engine_request = EngineRequest(
            request_id, prompt, params, max_tokens, tokenizer.end_tokens, queue_timeout=config.queue_timeout
        )
        out = OutputProcessor(tokenizer, stops(body), chat=False, logprobs=body.logprobs is not None)
        base = {"id": request_id, "created": int(time.time()), "model": body.model or config.model_name}
        events = run(engine_request, out)
        if body.stream:
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            _, events = await first_or_error(events)
            return sse(completion_chunks(events, base, include_usage))
        return JSONResponse(await until_disconnect(http, completion_whole(events, base)))

    async def completion_chunks(
        events: AsyncIterator[Delta | _Outcome], base: dict[str, Any], include_usage: bool
    ) -> AsyncIterator[dict[str, Any]]:
        chunk = {**base, "object": "text_completion"}
        offset = 0
        async for item in events:
            if isinstance(item, Delta):
                logprobs = _legacy_logprobs(item.logprobs, offset) if item.logprobs else None
                offset += len(item.content)
                yield {
                    **chunk,
                    "choices": [{"index": 0, "text": item.content, "logprobs": logprobs, "finish_reason": None}],
                }
            else:
                if item.error:
                    raise APIError(500, item.error, "server_error", "generation_failed")
                yield {**chunk, "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": item.reason}]}
                if include_usage:
                    yield {**chunk, "choices": [], "usage": item.usage}

    async def completion_whole(events: AsyncIterator[Delta | _Outcome], base: dict[str, Any]) -> dict[str, Any]:
        text, logprobs = [], []
        async for item in events:
            if isinstance(item, Delta):
                text.append(item.content)
                logprobs += item.logprobs
                continue
            if item.error:
                raise APIError(500, item.error, "server_error", "generation_failed")
            choice = {"index": 0, "text": "".join(text), "finish_reason": item.reason,
                      "logprobs": _legacy_logprobs(logprobs, 0) if logprobs else None}  # fmt: skip
            return {**base, "object": "text_completion", "choices": [choice], "usage": item.usage}
        raise APIError(500, "generation ended without a result", "server_error")

    return app


def _call(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}


def _legacy_logprobs(entries: list[dict[str, Any]], offset: int) -> dict[str, Any]:
    """OpenAI's completions-API logprobs layout from chat-style entries."""
    offsets, at = [], offset
    for e in entries:
        offsets.append(at)
        at += len(e["token"])
    return {
        "tokens": [e["token"] for e in entries],
        "token_logprobs": [e["logprob"] for e in entries],
        "top_logprobs": [{t["token"]: t["logprob"] for t in e["top_logprobs"]} for e in entries],
        "text_offset": offsets,
    }
