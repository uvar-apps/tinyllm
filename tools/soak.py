#!/usr/bin/env python3
"""Soak test: a mix of requests against a running server for a while, watching the server's memory.

    uv run python tools/soak.py --pid <server pid> --minutes 60

Requests alternate between streamed and whole chat completions, text completions, prompts of different lengths,
and streams abandoned after a few chunks (which the server must cancel). Every minute it prints the server's
resident memory; at the end it compares the last reading with the one after the warm-up period and exits
non-zero if memory grew by more than --max-growth-mb, or if any request failed.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import openai


def rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError(f"no VmRSS for process {pid}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--pid", type=int, required=True, help="the server process")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--warmup-minutes", type=float, default=5, help="memory baseline taken after this long")
    ap.add_argument("--max-growth-mb", type=float, default=64)
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.base_url, api_key="none", timeout=600, max_retries=0)
    model = client.models.list().data[0].id
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    filler = " ".join(f"word{i}" for i in range(2000))
    kinds = itertools.cycle(["chat", "stream", "completion", "abandon", "long"])
    start, next_report, baseline, failures, done = time.monotonic(), 0.0, None, 0, 0
    while (elapsed := time.monotonic() - start) < args.minutes * 60:
        kind = next(kinds)
        try:
            if kind == "chat":
                client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Name a colour."}], max_tokens=32, extra_body=off
                )
            elif kind == "stream":
                for _ in client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Count to 20."}],
                    max_tokens=64,
                    stream=True,
                    extra_body=off,
                ):
                    pass
            elif kind == "completion":
                client.completions.create(
                    model=model, prompt="Once upon a time", max_tokens=32, temperature=0.8, seed=done
                )
            elif kind == "abandon":
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Tell a story."}],
                    max_tokens=500,
                    stream=True,
                    extra_body=off,
                )
                for i, _ in enumerate(stream):
                    if i == 3:
                        stream.close()
                        break
            else:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": filler[: 2000 + 50 * (done % 40)]}],
                    max_tokens=8,
                    extra_body=off,
                )
            done += 1
        except Exception as e:  # count and go on: the soak is about staying up
            failures += 1
            print(f"request {done} ({kind}) failed: {type(e).__name__}: {e}", flush=True)
        if elapsed >= args.warmup_minutes * 60 and baseline is None:
            baseline = rss_mb(args.pid)
        if elapsed >= next_report:
            print(
                f"{elapsed / 60:5.1f} min  {done:5d} requests  {failures} failed  RSS {rss_mb(args.pid):8.1f} MB",
                flush=True,
            )
            next_report += 60
    final = rss_mb(args.pid)
    growth = final - (baseline if baseline is not None else final)
    print(
        f"done: {done} requests, {failures} failed; RSS after warm-up {baseline} MB, at the end {final:.1f} MB "
        f"({growth:+.1f} MB)"
    )
    return 1 if failures or growth > args.max_growth_mb else 0


if __name__ == "__main__":
    sys.exit(main())
