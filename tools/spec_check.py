#!/usr/bin/env python3
"""Speculative decoding with the MTP block against plain decoding: same tokens, and how much faster.

    DEV=KFD+AMD uv run python tools/spec_check.py ~/.cache/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf

For each prompt it generates --tokens tokens greedily twice through `UpstreamRunner`: one token per call with the
device sampler, then with draft-and-verify cycles of K drafts. It prints tokens per cycle, both speeds, and where
the two outputs first differ, with the plain run's top-2 logit margin there. --seeded repeats the comparison with
sampling (temperature 1, top-k 20, top-p 0.95, seed 7); both runs use the same uniform number at each position.
The verify computes what one-token decoding computes bit for bit (tinyllm.tg.speculative), so the outputs must be
identical, greedy or seeded; --spec-only skips the plain runs and measures speed only. Exits non-zero on any
difference.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain how a refrigerator works, step by step, for a curious ten year old.",
    "What are the main differences between TCP and UDP? Answer with a short table and a summary.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "A train leaves at 9:40 and arrives at 13:05. It stopped for 25 minutes. How long was it moving?",  # thinking
]
THINKING = {4}  # prompts answered with thinking on
GREEDY = [0.0, 1.0, 1.0, 0.0, 0.0]


def chat(text: str, thinking: bool = False) -> str:
    end = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{end}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-k", type=int, default=3, help="drafts per cycle")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--seeded", action="store_true")
    ap.add_argument("--spec-only", action="store_true", help="skip the plain runs (speed only, no comparison)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]  # progress shows up in a log file

    from tinyllm.engine.sampling import Sampler, SamplingParams
    from tinyllm.runner.upstream import UpstreamRunner
    from tinyllm.tg.tokenizer import GGUFTokenizer

    start = time.perf_counter()
    runner = UpstreamRunner(args.model, max_context=4096, wait="poll", speculative=args.k)
    tok = GGUFTokenizer(runner.kv)
    runner.warmup()
    print(f"loaded and compiled in {time.perf_counter() - start:.0f} s, K = {runner.speculative}")
    k = runner.speculative
    settings = [("greedy", SamplingParams(temperature=0.0))]
    if args.seeded:
        settings.append(("seeded", SamplingParams(temperature=1.0, top_k=20, top_p=0.95, seed=7)))
    failed = False
    totals = {"plain": [0, 0.0], "spec": [0, 0.0]}
    for name, params in settings:
        for i, text in enumerate(PROMPTS[: args.prompts]):
            ids = tok.encode(chat(text, i in THINKING))
            runs: dict[str, tuple[list[int], list[np.ndarray] | None]] = {}
            cycles = 0
            for mode in ("spec",) if args.spec_only else ("plain", "spec"):
                runner.begin(ids)
                sampler = Sampler(params, ids)
                first = sampler.sample(runner.feed(ids)).token
                out, margins = [first], []
                t0 = time.perf_counter()
                while len(out) < args.tokens:
                    if mode == "plain":
                        o = runner.feed_sampled(out[-1], sampler.device_rows(1)[0])
                        vals = o[2:4]
                        margins.append(float(vals[0] - vals[1]))
                        out.append(sampler.from_device(o).token)
                    else:
                        rows, n = runner.feed_speculative(out[-1], sampler.device_rows(k + 1))
                        out += [sampler.from_device(rows[j]).token for j in range(n + 1)]
                        cycles += 1
                seconds = time.perf_counter() - t0
                out = out[: args.tokens]
                totals[mode][0] += len(out) - 1
                totals[mode][1] += seconds
                runs[mode] = (out, margins)
                print(f"  {name} {mode:5s}: {(len(out) - 1) / seconds:5.1f} tok/s", end="")
            spec = runs["spec"][0]
            plain, margins = runs.get("plain", (spec, []))
            diff = next((i for i, (a, b) in enumerate(zip(plain, spec, strict=True)) if a != b), None)
            per_cycle = (len(spec) - 1) / max(1, cycles)
            if diff is None:
                verdict = "identical"
            else:
                margin = margins[diff - 1] if diff >= 1 and margins else float("nan")
                verdict = f"DIFFER at token {diff} (plain top-2 margin {margin:.3f})"
                failed = True
            print(f"  {per_cycle:.2f} tokens/cycle  {verdict}  {tok.decode(spec[:16])!r}")
    for mode, (n, s) in totals.items():
        if n:
            print(f"{mode}: {n / s:.1f} tok/s over {n} tokens")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
