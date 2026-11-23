#!/usr/bin/env python3
"""Fixed-length prefill chunks (TINYLLM_STATIC_CHUNKS) against variable-length ones, on the GPU.

    uv run python tools/static_chunk_check.py <model.gguf> [--chunk 16] [--prompt 40] [--decode 4]

Prefills the same prompt twice in one process, once with every chunk through the variable-length program and once
with full chunks through the fixed-length one (the last, partial chunk still goes through the variable-length
program), then decodes greedily. Prints the largest logit difference after the prompt and after each decode step,
and exits non-zero if the greedy tokens differ or a difference exceeds --tolerance.

The two programs are compiled differently (fusion, tiling and so reduction order), so logits differ by rounding:
up to 0.24 on Qwen3.5-0.8B, with the same tokens and the same agreement with llama.cpp on Qwen3.8-27B (research
findings, section 5.5). The default tolerance catches errors such as wrong positions or state, not rounding.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=40, help="prompt tokens; not a multiple of --chunk covers both paths")
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=1.0, help="largest allowed logit difference")
    args = ap.parse_args()

    from tinyllm import device

    device.configure("amd")
    import numpy as np

    from tinyllm.tg.upstream import UpstreamSession
    from tinyllm.weights.resolve import find_model

    model = str(find_model(args.model))
    prompt = [1000 + (7 * i) % 3000 for i in range(args.prompt)]
    runs = []
    for static in (False, True):
        session = UpstreamSession(model, 1024, rows=1, chunk=args.chunk, logits=True, static_chunks=static)
        logits = [session.prefill(prompt)]
        for _ in range(args.decode):
            logits.append(session.step([int(np.argmax(logits[-1], axis=-1)[0])]))
        runs.append([np.asarray(x, dtype=np.float64) for x in logits])
        del session
    failed = False
    for i, (a, b) in enumerate(zip(*runs, strict=True)):
        diff = float(np.max(np.abs(a - b)))
        same = bool(np.argmax(a, -1)[0] == np.argmax(b, -1)[0])
        bad = diff > args.tolerance or not same
        failed |= bad
        what = "after prompt" if i == 0 else f"decode step {i}"
        print(f"{what:14s} largest logit difference {diff:.4f}, same greedy token: {same}{'  FAIL' if bad else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
