#!/usr/bin/env python3
"""How often Qwen3.8's MTP block predicts the main model's next token, and what a T-token main call costs.

    DEV=KFD+AMD uv run python tools/mtp_check.py ~/.cache/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf

For each prompt the main model decodes greedily one token per call, recording its output-normed hidden at every
position. The MTP block then runs over the whole sequence, row i pairing hidden i with token i + 1, and its argmax
at row i is compared with token i + 2: the first draft's acceptance rate under greedy verification, which is what
the speculative loop sees when the MTP's KV holds the verified rows. With --timing it also times main-model calls
of 1 to 8 tokens with logits for every row, which is the verify step's cost.
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
    ("A train leaves at 9:40 and arrives at 13:05. It stopped for 25 minutes. How long was it moving?", True),
]


def chat(text: str, thinking: bool) -> str:
    end = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{end}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=160, help="tokens to generate per prompt")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()

    from tinyllm import tg
    from tinyllm.tg import Tensor, TinyJit, UOp, dtypes, mtp
    from tinyllm.tg import kernels as tl_kernels
    from tinyllm.tg.tokenizer import GGUFTokenizer

    tg.set_polling_wait()
    tl_kernels.install()
    start = time.perf_counter()
    model, kv, block = mtp.load(args.model, 4096, realize=False)
    if block is None:
        print("this GGUF has no MTP block")
        return 1
    print(f"loaded in {time.perf_counter() - start:.1f} s")
    tok = GGUFTokenizer(kv)
    v_pos = UOp.variable("start_pos_0", 0, model.max_context - 1)

    def forward_h(tokens: Tensor, *start_pos: UOp) -> tuple[Tensor, Tensor]:
        x = model.token_embd(tokens).float()
        idx_mask = None
        for b in model.blk:
            x, idx_mask = b(x, start_pos, idx_mask, tokens)
        h = model.output_norm(x)
        logits, h = model.output(h).float()[0].contiguous(), h.float().contiguous()
        Tensor.realize(logits, h)
        return logits, h

    def mtp_logits(tokens: Tensor, h: Tensor, *start_pos: UOp) -> Tensor:
        return model.output(block.hidden(model, tokens, h, start_pos)).float()[0].realize()

    main_jit: dict[int, TinyJit] = {}
    rows = 16
    mtp_jit = TinyJit(mtp_logits)

    def run_main(ids: list[int], pos: int) -> tuple[np.ndarray, np.ndarray]:
        jit = main_jit.setdefault(len(ids), TinyJit(forward_h))
        logits, h = jit(Tensor([ids], dtype=dtypes.int32).realize(), v_pos.bind(pos))
        return logits.numpy(), h.numpy()[0]

    total = agree = 0
    for text, thinking in PROMPTS[: args.prompts]:
        ids = tok.encode(chat(text, thinking))
        hs: list[np.ndarray] = []
        seq = list(ids)
        t0 = time.perf_counter()
        for pos in range(len(ids) + args.tokens - 1):
            logits, h = run_main([seq[pos]], pos)
            hs.append(h[0])
            if pos >= len(ids) - 1:
                seq.append(int(np.argmax(logits[0])))
        gen_s = time.perf_counter() - t0
        # MTP rows i = 0 .. L-2: (hidden i, token i+1) -> predicts token i+2
        n = len(seq) - 1
        preds: list[int] = []
        for lo in range(0, n, rows):
            hi = min(lo + rows, n)
            t = np.zeros((1, rows), dtype=np.int32)
            hh = np.zeros((1, rows, hs[0].size), dtype=np.float32)
            t[0, : hi - lo] = seq[lo + 1 : hi + 1]
            hh[0, : hi - lo] = np.stack(hs[lo:hi])
            out = mtp_jit(Tensor(t).realize(), Tensor(hh).realize(), v_pos.bind(lo)).numpy()
            preds += [int(np.argmax(out[r])) for r in range(hi - lo)]
        # generated region: token i+2 was chosen by the main model for i + 2 >= len(ids)
        first = len(ids) - 2
        hits = [preds[i] == seq[i + 2] for i in range(first, len(seq) - 2)]
        prompt_hits = [preds[i] == seq[i + 2] for i in range(0, first)]
        total += len(hits)
        agree += sum(hits)
        rate = sum(hits) / max(1, len(hits))
        teacher = sum(prompt_hits) / max(1, len(prompt_hits))
        print(
            f"{len(ids):4d} prompt + {args.tokens} tokens in {gen_s:.1f} s: MTP first-draft acceptance {rate:.3f} "
            f"(on the prompt text {teacher:.3f})  {tok.decode(seq[len(ids) : len(ids) + 24])!r}"
        )
    print(f"overall first-draft acceptance {agree / max(1, total):.3f} over {total} tokens")

    if args.timing:
        for t in range(1, 9):
            ids = [1] * t
            for _ in range(3):
                run_main(ids, 100)
            tg.synchronize()
            jit = main_jit[t]
            x = Tensor([ids], dtype=dtypes.int32).realize()
            start = time.perf_counter()
            for _ in range(8):
                out = jit(x, v_pos.bind(100))
            tg.synchronize()
            del out
            print(f"main call, {t} token(s), logits for every row: {(time.perf_counter() - start) / 8 * 1e3:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
