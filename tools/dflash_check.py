#!/usr/bin/env python3
"""How many of DFlash2's drafts the target would accept, and what one draft block costs.

    DEV=KFD+AMD uv run python tools/dflash_check.py ~/.cache/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf

The target decodes greedily one token per call and hands its layer inputs at DFlash2's target layers to the
drafter after every token (injection), as the speculative loop does. Before feeding each token the drafter
proposes a block from the context so far; afterwards the block's drafts are compared with the tokens the target
went on to choose. Reports the mean number of leading matches per block (the tokens a verify step would accept,
plus one) per prompt, next to the MTP's figures in docs/01-research-findings.md, section 5.7.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

PROMPTS = [
    ("Write a Python function that parses an ISO 8601 date string without using datetime, with tests.", False),
    ("Explain how a refrigerator works, step by step, for a curious ten year old.", False),
    ("What are the main differences between TCP and UDP? Answer with a short table and a summary.", False),
    ("Write a short story about a lighthouse keeper who finds a message in a bottle.", False),
    ("A train leaves at 9:40 and arrives at 13:05. It stopped for 25 minutes. How long was it moving?", True),
]
DRAFTER = "~/.cache/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"


def chat(text: str, thinking: bool) -> str:
    end = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{end}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--drafter", default=DRAFTER)
    ap.add_argument("--tokens", type=int, default=160, help="tokens to generate per prompt")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from pathlib import Path

    from tinygrad.llm.model import Transformer

    from tinyllm import tg
    from tinyllm.tg import Tensor, TinyJit, UOp, dtypes
    from tinyllm.tg import kernels as tl_kernels
    from tinyllm.tg.dflash import DFlash
    from tinyllm.tg.tokenizer import GGUFTokenizer

    tg.set_polling_wait()
    tl_kernels.install()
    start = time.perf_counter()
    model, kv = Transformer.from_gguf(args.model, 4096, realize=False)
    drafter = DFlash.load(Path(args.drafter).expanduser(), 4096)
    print(
        f"loaded in {time.perf_counter() - start:.1f} s; target layers {drafter.target_layers}, "
        f"block {drafter.block_size}, mask {drafter.mask_token}"
    )
    tok = GGUFTokenizer(kv)
    v_pos = UOp.variable("start_pos_0", 0, model.max_context - 1)
    taps = set(drafter.target_layers)
    block = drafter.block_size

    def forward(tokens: Tensor, *start_pos: UOp) -> tuple[Tensor, Tensor]:
        x = model.token_embd(tokens).float()
        features = []
        for i, b in enumerate(model.blk):
            if i in taps:
                features.append(x)
            x, _ = b(x, start_pos, None, tokens)
        logits = model.output(model.output_norm(x)).float()[0, -1].contiguous()
        stores = drafter.inject(features[0].cat(*features[1:], dim=-1), start_pos[0])
        token = topk_id(logits)
        Tensor.realize(token, Tensor(drafter.cache[0].uop.after(*stores)))
        return token

    def topk_id(logits: Tensor) -> Tensor:
        from tinyllm.tg.kernels.topk import topk

        return topk(logits, 1)[1].contiguous()

    def draft(tokens: Tensor, pos: UOp) -> Tensor:
        h = drafter.hidden(model, tokens, pos)[0, 1:]
        return drafter.select(h, model.output(h).float(), tokens[0, :1]).contiguous().realize()

    forward_jit, draft_jit = TinyJit(forward), TinyJit(draft)
    all_accepted: list[int] = []
    draft_seconds: list[float] = []
    for text, thinking in PROMPTS[: args.prompts]:
        ids = tok.encode(chat(text, thinking))
        seq, blocks = list(ids), {}
        for pos in range(len(ids) + args.tokens - 1):
            if pos >= len(ids):  # the token at pos was sampled, not yet fed: draft the block that follows it
                t0 = time.perf_counter()
                row = np.full((1, block), drafter.mask_token, dtype=np.int32)
                row[0, 0] = seq[pos]
                drafts = draft_jit(Tensor(row).realize(), v_pos.bind(pos)).numpy()[0]
                draft_seconds.append(time.perf_counter() - t0)
                blocks[pos] = [int(d) for d in drafts]
            token = forward_jit(Tensor([[seq[pos]]], dtype=dtypes.int32).realize(), v_pos.bind(pos)).numpy()
            if pos >= len(ids) - 1:
                seq.append(int(token[0]))
        accepted = []
        for pos, drafts in blocks.items():
            future = seq[pos + 1 : pos + 1 + len(drafts)]
            n = 0
            while n < len(future) and drafts[n] == future[n]:
                n += 1
            if len(future) == len(drafts):
                accepted.append(n)
        all_accepted += accepted
        print(
            f"{len(ids):4d} prompt + {args.tokens}: {1 + np.mean(accepted):.2f} tokens per block "
            f"(accepted drafts {np.mean(accepted):.2f} of {block - 1})  {tok.decode(seq[len(ids) : len(ids) + 20])!r}"
        )
    times = sorted(draft_seconds[5:])
    print(
        f"overall {1 + np.mean(all_accepted):.2f} tokens per block over {len(all_accepted)} blocks; "
        f"one draft block {times[len(times) // 2] * 1e3:.1f} ms (median, with host round trip)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
