#!/usr/bin/env python3
"""Phase 1 exit checks against a running server, through the OpenAI Python SDK.

    uv run tinyllm serve Qwen3.8-27B-UD-Q4_K_XL.gguf &          # then:
    uv run python tools/openai_sdk_check.py --base-url http://127.0.0.1:8000/v1
    uv run python tools/openai_sdk_check.py --skip tools,speed  # e.g. for a small model

Checks: streaming chat, reasoning on and off, stop sequences, seeded reproducibility, a tool-call round trip, two
clients at once, and speed (decode tokens per second from the stream, prefill tokens per second from a long
prompt's time to first token). Prints one line per check and exits non-zero if any fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

import openai

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
THINKING = {"chat_template_kwargs": {"enable_thinking": True}}  # some templates (Qwen3.5) default to off


def extra(obj: Any, name: str) -> Any:
    return (getattr(obj, "model_extra", None) or {}).get(name)


def stream_chat(client: openai.OpenAI, **kwargs: Any) -> dict[str, Any]:
    start = time.perf_counter()
    content, reasoning, finish, usage, stamps = "", "", None, None, []
    for chunk in client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **kwargs):
        if chunk.usage is not None:
            usage = chunk.usage
        for choice in chunk.choices:
            piece = (choice.delta.content or "") + (extra(choice.delta, "reasoning_content") or "")
            if piece:
                stamps.append(time.perf_counter())
            content += choice.delta.content or ""
            reasoning += extra(choice.delta, "reasoning_content") or ""
            finish = choice.finish_reason or finish
    return {
        "content": content,
        "reasoning": reasoning,
        "finish": finish,
        "usage": usage,
        "start": start,
        "stamps": stamps,
    }


def check_stream(client: openai.OpenAI, model: str) -> str:
    r = stream_chat(
        client,
        model=model,
        messages=[{"role": "user", "content": "Name three primary colours."}],
        max_tokens=512,
        extra_body=NO_THINKING,
    )
    assert r["content"].strip(), "no content"
    assert r["finish"] in ("stop", "length"), r["finish"]
    assert r["usage"] is not None and r["usage"].completion_tokens > 0, "no usage"
    return f"{r['usage'].completion_tokens} tokens, finish {r['finish']}: {r['content'].strip()[:60]!r}"


def check_reasoning(client: openai.OpenAI, model: str) -> str:
    question = [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]
    on = client.chat.completions.create(model=model, messages=question, max_tokens=2048, extra_body=THINKING)
    off = client.chat.completions.create(model=model, messages=question, max_tokens=256, extra_body=NO_THINKING)
    thought = extra(on.choices[0].message, "reasoning_content") or ""
    assert thought.strip(), "thinking on: no reasoning_content"
    assert not (extra(off.choices[0].message, "reasoning_content") or "").strip(), "thinking off: reasoning_content"
    assert "<think>" not in (on.choices[0].message.content or ""), "a think tag leaked into the content"
    answer = (off.choices[0].message.content or "").strip()
    on_answer = (on.choices[0].message.content or "").strip()
    return f"on: {len(thought)} chars of reasoning, answer {on_answer[:20]!r}; off: {answer[:20]!r}"


def check_stop(client: openai.OpenAI, model: str) -> str:
    r = client.completions.create(model=model, prompt="1, 2, 3, 4, 5, 6,", max_tokens=40, temperature=0, stop=[" 9"])
    text = r.choices[0].text
    assert " 9" not in text and r.choices[0].finish_reason == "stop", (text, r.choices[0].finish_reason)
    chat = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 10, separated by commas."}],
        stop=["7"],
        max_tokens=200,
        temperature=0,
        extra_body=NO_THINKING,
    )
    content = chat.choices[0].message.content or ""
    assert "7" not in content and chat.choices[0].finish_reason == "stop", (content, chat.choices[0].finish_reason)
    return f"completion {text!r}; chat {content.strip()!r}"


def check_seed(client: openai.OpenAI, model: str) -> str:
    def ask(seed: int) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write a one-line poem about the sea."}],
            temperature=1.0,
            seed=seed,
            max_tokens=40,
            extra_body=NO_THINKING,
        )
        return r.choices[0].message.content or ""

    a, b, c = ask(42), ask(42), ask(7)
    assert a == b, f"seed 42 gave two different answers: {a!r} / {b!r}"
    return f"same text for the same seed; {'different' if a != c else 'same (unlikely)'} for another"


def check_tools(client: openai.OpenAI, model: str) -> str:
    # greedy: this checks the server's tool-call round trip, and sampled, a small model sometimes answers
    # without calling the tool (the 0.8B did once in five runs)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    r = client.chat.completions.create(model=model, messages=messages, tools=[WEATHER], max_tokens=1024, temperature=0)
    choice = r.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.tool_calls, (choice.finish_reason, choice.message)
    call = choice.message.tool_calls[0]
    args = json.loads(call.function.arguments)
    assert call.function.name == "get_weather" and "paris" in str(args.get("city", "")).lower(), call.function
    messages += [
        {"role": "assistant", "content": choice.message.content, "tool_calls": [call.model_dump()]},
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({"city": "Paris", "sky": "sunny", "celsius": 21}),
        },
    ]
    final = client.chat.completions.create(
        model=model, messages=messages, tools=[WEATHER], max_tokens=1024, temperature=0
    )
    answer = final.choices[0].message.content or ""
    assert final.choices[0].finish_reason == "stop" and "21" in answer, (final.choices[0].finish_reason, answer)
    return f"called {call.function.name}({call.function.arguments}); answer {answer.strip()[:60]!r}"


def check_concurrent(client: openai.OpenAI, model: str) -> str:
    """Two clients at once: both are served (one after the other, phase 1 runs one sequence at a time)."""
    import threading

    results: dict[int, Any] = {}

    def ask(i: int) -> None:
        try:
            results[i] = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Say the word number {i}."}],
                max_tokens=64,
                extra_body=NO_THINKING,
            )
        except Exception as e:  # reported below
            results[i] = e

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errors = [r for r in results.values() if isinstance(r, Exception)]
    assert not errors, errors
    assert all(r.choices[0].message.content for r in results.values()), "empty answer"
    return "both answered: " + " / ".join(
        repr((r.choices[0].message.content or "").strip()[:30]) for r in results.values()
    )


def check_speed(client: openai.OpenAI, model: str, min_decode: float, min_prefill: float) -> str:
    r = stream_chat(
        client,
        model=model,
        messages=[{"role": "user", "content": "Tell me a long story about a lighthouse."}],
        max_tokens=300,
        extra_body=NO_THINKING,
    )
    stamps = r["stamps"]
    decode = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 10 else 0.0
    words = " ".join(f"item{i}" for i in range(1500))
    long = stream_chat(
        client,
        model=model,
        messages=[{"role": "user", "content": f"Here is a list: {words}. How many items?"}],
        max_tokens=8,
        extra_body=NO_THINKING,
    )
    ttft = long["stamps"][0] - long["start"] if long["stamps"] else float("inf")
    prompt = long["usage"].prompt_tokens if long["usage"] else 0
    prefill = prompt / ttft
    summary = f"decode {decode:.2f} tok/s (stream chunks), prefill {prefill:.0f} tok/s"
    summary += f" ({prompt} tokens, first token {ttft:.2f}s)"
    assert decode >= min_decode and prefill >= min_prefill, summary
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="none")
    ap.add_argument(
        "--skip", default="", help="comma-separated checks to skip: stream,reasoning,stop,seed,tools,concurrent,speed"
    )
    ap.add_argument("--min-decode", type=float, default=12.3, help="decode tok/s required (default: %(default)s)")
    ap.add_argument("--min-prefill", type=float, default=270, help="prefill tok/s required (default: %(default)s)")
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600, max_retries=0)
    model = client.models.list().data[0].id
    print(f"model {model}")
    checks: dict[str, Callable[[], str]] = {
        "stream": lambda: check_stream(client, model),
        "reasoning": lambda: check_reasoning(client, model),
        "stop": lambda: check_stop(client, model),
        "seed": lambda: check_seed(client, model),
        "tools": lambda: check_tools(client, model),
        "concurrent": lambda: check_concurrent(client, model),
        "speed": lambda: check_speed(client, model, args.min_decode, args.min_prefill),
    }
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    failed = 0
    for name, check in checks.items():
        if name in skip:
            continue
        try:
            print(f"PASS {name:10s} {check()}", flush=True)
        except Exception as e:  # report every check, then fail
            failed += 1
            print(f"FAIL {name:10s} {type(e).__name__}: {e}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
