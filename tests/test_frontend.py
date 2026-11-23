"""Frontend on the CPU: chat template, tokenizer adapter, detokenizer with stop strings, reasoning and tool routing."""

from __future__ import annotations

import json

import pytest
from conftest import _find

from tinyllm.frontend.detokenizer import Detokenizer
from tinyllm.frontend.router import StreamRouter, parse_tool_call
from tinyllm.frontend.template import ChatTemplate, TemplateRenderError, normalize_messages

TARGET = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "zip": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
}


@pytest.fixture(scope="module")
def qwen38_template() -> ChatTemplate:
    from tinyllm.weights import gguf

    path = _find(TARGET)
    if path is None:
        pytest.skip(f"{TARGET} not found")
    return ChatTemplate(str(gguf.read(path, header_only=True).kv["tokenizer.chat_template"]))


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    from tinyllm.tg.tokenizer import GGUFTokenizer
    from tinyllm.weights import gguf

    path = _find("Qwen3.5-0.8B-Q8_0.gguf")
    if path is None:
        pytest.skip("development model not found")
    return GGUFTokenizer(gguf.read(path, max_array_items=1 << 22, header_only=True).kv)


def test_qwen38_template_thinking_on_and_off(qwen38_template: ChatTemplate) -> None:
    messages = [{"role": "user", "content": "Hi"}]
    on = qwen38_template.render(messages)
    off = qwen38_template.render(messages, enable_thinking=False)
    assert on.endswith("<|im_start|>assistant\n<think>\n")
    assert off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<|im_start|>user\nHi<|im_end|>" in on


def test_qwen38_template_with_tools_and_a_round_trip(qwen38_template: ChatTemplate) -> None:
    messages = [
        {"role": "user", "content": "Weather in Rome?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]
    text = qwen38_template.render(messages, tools=[WEATHER])
    assert "<tools>" in text and '"name": "get_weather"' in text
    assert "<function=get_weather>\n<parameter=city>\nRome\n</parameter>" in text  # JSON arguments became an object
    assert "<tool_response>\nsunny\n</tool_response>" in text


def test_template_errors_and_the_sandbox() -> None:
    with pytest.raises(TemplateRenderError, match="no"):
        ChatTemplate("{{ raise_exception('no') }}").render([{"role": "user", "content": "x"}])
    with pytest.raises(TemplateRenderError):
        ChatTemplate("{{ messages.__class__.__mro__ }}").render([{"role": "user", "content": "x"}])
    with pytest.raises(TemplateRenderError):  # the immutable sandbox refuses to modify the request
        ChatTemplate("{% set _ = messages.append(1) %}").render([{"role": "user", "content": "x"}])


def test_normalize_messages_copies() -> None:
    original = [{"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}]
    out = normalize_messages(original)
    assert out[0]["content"] == "" and out[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert original[0]["content"] is None


def test_tokenizer_round_trip_and_end_tokens(tokenizer) -> None:  # type: ignore[no-untyped-def]
    text = "Hello, 世界! <|im_start|>user\ncode: x = 1<|im_end|>"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.special_tokens["<|im_start|>"] in ids
    assert tokenizer.is_end(tokenizer.special_tokens["<|im_end|>"])
    assert tokenizer.is_end(tokenizer.special_tokens["<|endoftext|>"])
    assert not tokenizer.is_end(ids[0])
    assert b"".join(tokenizer.token_bytes(t) for t in ids).decode() == text


def _bytes_table(pieces: list[bytes]):  # type: ignore[no-untyped-def]
    return lambda token: pieces[token]


def test_detokenizer_joins_split_characters() -> None:
    pieces = [b"caf", "é".encode()[:1], "é".encode()[1:], b"!"]
    d = Detokenizer(_bytes_table(pieces))
    assert "".join(d.push(t) for t in range(4)) + d.finish() == "café!"
    assert Detokenizer(_bytes_table(pieces)).push(1) == ""  # half a character is held


def test_detokenizer_stops_and_never_emits_a_partial_stop() -> None:
    pieces = [b"one ", b"two ", b"ST", b"OP", b" three"]
    d = Detokenizer(_bytes_table(pieces), stop=["STOP"])
    out = [d.push(t) for t in range(5)]
    assert out == ["one ", "two ", "", "", ""] and d.stopped == "STOP"
    # a held prefix that turns out not to be a stop string is released
    d = Detokenizer(_bytes_table([b"a S", b"Tx", b" end"]), stop=["STOP"])
    assert "".join(d.push(t) for t in range(3)) + d.finish() == "a STx end" and d.stopped is None


def _route(router: StreamRouter, pieces: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, piece in enumerate([*pieces, ""]):
        for field, text in router.route(piece, final=i == len(pieces)):
            out[field] = out.get(field, "") + text
    return out


def test_router_reasoning_then_content() -> None:
    # the prompt ended with "<think>\n": output starts inside the block
    got = _route(StreamRouter(reasoning=True), ["Let me ", "think.</th", "ink>\n\nThe answer", " is 4."])
    assert got == {"reasoning_content": "Let me think.", "content": "The answer is 4."}
    got = _route(StreamRouter(), ["<thi", "nk>hmm</think>", "ok"])
    assert got == {"reasoning_content": "hmm", "content": "ok"}
    assert _route(StreamRouter(), ["<b>bold</b>"]) == {"content": "<b>bold</b>"}


def test_router_tool_calls_with_schema_types() -> None:
    router = StreamRouter(tools=[WEATHER])
    text = (
        "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n"
        "<parameter=zip>\n02134\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    )
    got = _route(router, [text[:20], text[20:47], text[47:]])
    assert got == {"content": "Checking.\n\n"}
    calls, unparsed = router.finish()
    assert unparsed == "" and len(calls) == 1 and calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "New York", "zip": "02134", "days": 3}


def test_router_json_calls_failed_parses_and_no_tools() -> None:
    router = StreamRouter(tools=[WEATHER])
    _route(router, ['<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>'])
    calls, _ = router.finish()
    assert calls[0].name == "get_weather" and json.loads(calls[0].arguments) == {"city": "Oslo"}
    router = StreamRouter(tools=[WEATHER])
    _route(router, ["<tool_call>not a call</tool_call>"])
    assert router.finish() == ([], "<tool_call>not a call</tool_call>")
    # without tools in the request the text is content
    assert _route(StreamRouter(), ["a <tool_call>x</tool_call>"]) == {"content": "a <tool_call>x</tool_call>"}


def test_parse_tool_call_rejects_text_outside_parameters() -> None:
    assert parse_tool_call("<function=f>\n<parameter=a>\n1\n</parameter>\n</function>") == ("f", {"a": 1})
    assert parse_tool_call("<function=f>\nstray<parameter=a>\n1\n</parameter>\n</function>") is None
    assert parse_tool_call("plain words") is None
