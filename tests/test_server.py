"""The HTTP server through the OpenAI Python SDK, with a scripted runner on the CPU."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import openai
import pytest
from fakes import ByteTokenizer, ScriptRunner
from fastapi.testclient import TestClient

from tinyllm.engine.core import Engine
from tinyllm.frontend.template import ChatTemplate
from tinyllm.server.app import ServerConfig, create_app
from tinyllm.server.client import EngineClient

TOK = ByteTokenizer()
END = 257
WEATHER = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
}


@dataclass
class Server:
    runner: ScriptRunner
    engine: Engine
    http: TestClient
    sdk: openai.OpenAI


def _script(text: str, end: bool = True) -> list[int]:
    return TOK.encode(text) + ([END] if end else [])


@pytest.fixture
def make_server() -> Iterator:  # type: ignore[type-arg]
    started: list[tuple[Engine, threading.Thread, TestClient]] = []

    def make(script: list[int], api_key: str | None = None, max_waiting: int = 4, noise: float = 0.0,
             max_context: int = 1 << 20) -> Server:  # fmt: skip
        runner = ScriptRunner(script, noise=noise, max_context=max_context)
        engine = Engine(runner, max_waiting=max_waiting, prefill_slice=1 << 20)
        thread = threading.Thread(target=engine.run, daemon=True)
        thread.start()
        config = ServerConfig("test-model", runner.max_context, {"temperature": 0.0}, api_key=api_key)
        app = create_app(EngineClient(engine), TOK, ChatTemplate(str(TOK.chat_template)), config)
        http = TestClient(app)
        started.append((engine, thread, http))
        sdk = openai.OpenAI(base_url="http://testserver/v1", api_key=api_key or "none", http_client=http, max_retries=0)
        return Server(runner, engine, http, sdk)

    yield make
    for engine, thread, http in started:
        http.close()
        engine.stop()
        thread.join(5)


def test_health_models_and_metrics(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("hi"))
    assert s.http.get("/health").json()["status"] == "ok"
    assert [m.id for m in s.sdk.models.list()] == ["test-model"]
    assert "tinyllm_requests_running" in s.http.get("/metrics").text


def test_chat_with_reasoning(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("Let me think.</think>\n\nHello!"))
    r = s.sdk.chat.completions.create(model="test-model", messages=[{"role": "user", "content": "hi"}])
    message = r.choices[0].message
    assert message.content == "Hello!" and message.model_extra["reasoning_content"] == "Let me think."
    assert r.choices[0].finish_reason == "stop"
    prompt = TOK.decode(s.runner.prompts[-1])
    assert prompt.endswith("<|im_start|>assistant\n<think>\n") and r.usage.prompt_tokens == len(s.runner.prompts[-1])
    assert r.usage.completion_tokens == len(_script("Let me think.</think>\n\nHello!"))


def test_chat_streaming_with_usage(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("Plan.</think>\n\nAnswer."))
    stream = s.sdk.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "q"}], stream=True, stream_options={"include_usage": True}
    )
    reasoning, content, finish, usage = "", "", None, None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            reasoning += (choice.delta.model_extra or {}).get("reasoning_content") or ""
            content += choice.delta.content or ""
            finish = choice.finish_reason or finish
    assert (reasoning, content, finish) == ("Plan.", "Answer.", "stop") and usage is not None


def test_thinking_off(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("Hello!"))
    r = s.sdk.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert r.choices[0].message.content == "Hello!" and "reasoning_content" not in (
        r.choices[0].message.model_extra or {}
    )
    assert TOK.decode(s.runner.prompts[-1]).endswith("<think>\n\n</think>\n\n")


def test_tool_call_round_trip(make_server) -> None:  # type: ignore[no-untyped-def]
    call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>"
    s = make_server(_script(f"I will check.</think>\n\n{call}"))
    r = s.sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "weather?"}], tools=[WEATHER])
    choice = r.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content is None
    (tool_call,) = choice.message.tool_calls
    assert tool_call.function.name == "get_weather" and json.loads(tool_call.function.arguments) == {"city": "Rome"}
    # the result goes back as a tool message; the template renders the round trip
    s.runner.script = _script("It is sunny.</think>\n\nSunny in Rome.")
    follow = s.sdk.chat.completions.create(
        model="m",
        tools=[WEATHER],
        messages=[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [tool_call.model_dump()]},
            {"role": "tool", "tool_call_id": tool_call.id, "content": "sunny"},
        ],
    )
    assert follow.choices[0].message.content == "Sunny in Rome."
    assert "<|im_start|>tool\nsunny<|im_end|>" in TOK.decode(s.runner.prompts[-1])


def test_tool_call_streaming(make_server) -> None:  # type: ignore[no-untyped-def]
    call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"
    s = make_server(_script(f"x</think>\n\n{call}"))
    stream = s.sdk.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "w"}], tools=[WEATHER], stream=True
    )
    calls, finish = [], None
    for chunk in stream:
        for choice in chunk.choices:
            calls += choice.delta.tool_calls or []
            finish = choice.finish_reason or finish
    assert finish == "tool_calls" and calls[0].function.name == "get_weather"


def test_stop_strings_and_length(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("x</think>\n\nabc STOP def"))
    r = s.sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stop=["STOP"])
    assert r.choices[0].message.content == "abc " and r.choices[0].finish_reason == "stop"
    s.runner.script = _script("x</think>\n\n0123456789", end=False)
    r = s.sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], max_tokens=6)
    assert r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 6


def test_completions_endpoint_with_logprobs(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script(" world"))
    r = s.sdk.completions.create(model="m", prompt="hello", max_tokens=20, logprobs=2)
    assert r.choices[0].text == " world" and r.choices[0].finish_reason == "stop"
    lp = r.choices[0].logprobs
    assert lp.tokens == list(" world") and len(lp.top_logprobs) == 6 and lp.text_offset[1] == 1
    assert TOK.decode(s.runner.prompts[-1]) == "hello"  # no chat template on this endpoint
    stream = s.sdk.completions.create(model="m", prompt="hello", max_tokens=20, stream=True)
    assert "".join(c.choices[0].text for c in stream if c.choices) == " world"


def test_chat_logprobs(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("x</think>\n\nok"))
    r = s.sdk.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "q"}], logprobs=True, top_logprobs=3
    )
    entries = r.choices[0].logprobs.content
    assert [e.token for e in entries][-2:] == ["o", "k"] and len(entries[0].top_logprobs) == 3


def test_seed_reproduces_sampled_text(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("abc"), noise=3.0)

    def ask(seed: int) -> str:
        r = s.sdk.completions.create(model="m", prompt="p", max_tokens=12, temperature=1.0, seed=seed)
        return r.choices[0].text

    assert ask(1) == ask(1) and ask(1) != ask(2)


def test_errors(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("hi"), max_context=64)
    with pytest.raises(openai.BadRequestError, match="context"):
        s.sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "x" * 100}])
    with pytest.raises(openai.BadRequestError, match="n=1"):
        s.sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}], n=2)
    with pytest.raises(openai.BadRequestError, match="top_p"):
        s.sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}], top_p=0)
    bad = s.http.post("/v1/chat/completions", json={"model": "m", "messages": "not a list"})
    assert bad.status_code == 400 and bad.json()["error"]["type"] == "invalid_request_error"


def test_api_key(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("hi"), api_key="secret")
    assert s.sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}]).choices
    assert s.http.post("/v1/chat/completions", json={"messages": []}).status_code == 401
    assert s.http.get("/health").status_code == 200


def test_busy_queue_answers_429(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("hi"), max_waiting=0)
    hold = threading.Event()
    s.runner.hold = hold
    first = threading.Thread(target=lambda: s.http.post("/v1/completions", json={"prompt": "a", "max_tokens": 2}))
    first.start()
    for _ in range(200):  # wait until the first request occupies the engine
        if s.engine.load[0]:
            break
        threading.Event().wait(0.01)
    busy = s.http.post("/v1/completions", json={"prompt": "b", "max_tokens": 2})
    hold.set()
    first.join(5)
    assert busy.status_code == 429 and busy.json()["error"]["code"] == "queue_full"
    stream = s.http.post("/v1/completions", json={"prompt": "c", "max_tokens": 2, "stream": True})
    assert stream.status_code == 200


def test_disconnect_cancels_the_request(make_server) -> None:  # type: ignore[no-untyped-def]
    s = make_server(_script("x" * 50, end=False))
    with s.http.stream("POST", "/v1/completions", json={"prompt": "a", "max_tokens": 10_000, "stream": True}) as r:
        for line in r.iter_lines():
            if line.startswith("data: {"):
                break  # leave after the first chunk
    for _ in range(300):
        if s.engine.metrics.finished["cancelled"]:
            break
        threading.Event().wait(0.01)
    assert s.engine.metrics.finished["cancelled"] == 1
