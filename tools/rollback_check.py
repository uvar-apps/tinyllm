#!/usr/bin/env python3
"""The speculative verify's DeltaNet rollback against plain runs (tinyllm.tg.speculative), on any Qwen3.5/3.8 GGUF.

    DEV=KFD+AMD uv run python tools/rollback_check.py [model.gguf] [--rows 4]

After a short prompt it saves the recurrent and convolution state of every DeltaNet layer, then decodes the next
`rows` tokens one per call, keeping the logits and the state after each. A recorded verify call of all rows from
the saved state must give the same logits at every row and the same final state; its snapshot j must equal the
state after token j, and `restore_from(j)` must put that state back, all bit for bit (tinyllm.tg.speculative).
Exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None, help="default: the dev model (tests/conftest.py)")
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()

    from tinyllm import tg
    from tinyllm.tg import Tensor, UOp, dtypes
    from tinyllm.tg import speculative as spec
    from tinyllm.tg.upstream import UpstreamSession
    from tinyllm.weights.resolve import find_model

    path = find_model(args.model or "Qwen3.5-0.8B-Q8_0.gguf")
    session = UpstreamSession(path, max_context=1024, chunk=32, logits=True)
    model, rows = session.model, args.rows
    blocks = spec.recurrent_blocks(model)
    for b in blocks:
        b.allocate(rows, tg.default_device())
    verify_blocks = [*blocks, *spec.attention_blocks(model)]
    prompt = [(i * 7919) % 5000 + 100 for i in range(20)]
    following = [(i * 104729) % 5000 + 100 for i in range(rows)]
    session.prefill(prompt)
    start = len(prompt)
    v_pos = UOp.variable("start_pos", 0, model.max_context - 1)

    def states() -> list[tuple[np.ndarray, np.ndarray]]:
        return [(b.recurrent_state.numpy().copy(), b.conv_state.numpy().copy()) for b in blocks]

    def put(saved: list[tuple[np.ndarray, np.ndarray]]) -> None:
        writes = []
        for b, (r, c) in zip(blocks, saved, strict=True):
            writes += [b.recurrent_state.assign(Tensor(r)), b.conv_state.assign(Tensor(c))]
        Tensor.realize(*writes)

    def run(tokens: list[int], record: bool = False, at: int = start) -> np.ndarray:
        t = Tensor([tokens], dtype=dtypes.int32)
        x = model.token_embd(t).float()
        with spec.recording(verify_blocks) if record else _nothing():
            for block in model.blk:
                x, _ = block(x, (v_pos.bind(at),), None, t)
        return model.output(model.output_norm(x)).float()[0].numpy()

    base = states()
    # plain decoding: one token per call, the state after each
    single, plain_states = [], []
    for j, t in enumerate(following):
        single.append(run([t], at=start + j)[0])
        plain_states.append(states())
    plain_logits = np.stack(single)
    # not judged: an ordinary call of all rows at once (tinygrad's generic attention for a few rows)
    put(base)
    batched = run(following)
    drift = float(np.max(np.abs(batched - plain_logits)))
    same = int(np.sum(batched.argmax(-1) == plain_logits.argmax(-1)))
    print(
        f"info an ordinary {rows}-row call against one-token calls: max logit difference {drift:.3g}, "
        f"argmax {same}/{rows}"
    )
    put(base)
    recorded_logits = run(following, record=True)
    recorded_final = states()
    snapshots = [[s.numpy().copy() for s in b.snapshots] for b in blocks]

    failures = 0

    def check(name: str, a: np.ndarray, b: np.ndarray) -> None:
        nonlocal failures
        diff = float(np.max(np.abs(a - b)))
        ok = diff == 0  # the verify computes what decoding computes, bit for bit
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max difference {diff:.3g}")

    check("logits at every row against one-token calls", recorded_logits, plain_logits)
    for i, ((r, c), (pr, pc)) in enumerate(zip(recorded_final, plain_states[-1], strict=True)):
        if i == 0 or i == len(blocks) - 1:
            check(f"layer {i} final recurrent state", r, pr)
            check(f"layer {i} final convolution state", c, pc)
    for j in range(rows - 1):
        worst_r = max(float(np.max(np.abs(s[j] - p[0]))) for s, p in zip(snapshots, plain_states[j], strict=True))
        check(f"snapshot {j} (all layers)", np.asarray(worst_r), np.asarray(0.0))
        put(recorded_final)
        Tensor.realize(*(w for b in blocks for w in b.restore_from(j)))
        for i, ((r, c), (pr, pc)) in enumerate(zip(states(), plain_states[j], strict=True)):
            if i in (0, len(blocks) - 1):
                check(f"restore {j}: layer {i} recurrent state", r, pr)
                check(f"restore {j}: layer {i} convolution state", c, pc)
    print(f"tinyllm rollback check: {failures} failure(s) over {len(blocks)} DeltaNet layers, {rows} rows")
    return 1 if failures else 0


class _nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


if __name__ == "__main__":
    sys.exit(main())
