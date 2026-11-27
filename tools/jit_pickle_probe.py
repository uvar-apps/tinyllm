#!/usr/bin/env python3
"""Does a pickled tinygrad JIT remove the start-up warm-up?

    uv run python tools/jit_pickle_probe.py save <model.gguf> out/jit.pkl
    uv run python tools/jit_pickle_probe.py load out/jit.pkl

`save` warms a model up the normal way, records a short greedy continuation, and pickles the
two captured JIT programs (prefill and decode). `load` starts a fresh process, unpickles
them, and runs the same prompt without constructing the model at all.

The pickle embeds every buffer the programs touch, weights included, so this is only
practical as-is for a small model. It answered the question of principle. The server keeps
its programs between starts with `tinyllm.tg.programs`, which binds them to the buffers of
the loaded model instead (`tools/program_cache_check.py` checks that).
"""

from __future__ import annotations

import pickle
import sys
import time

PROCESS_START = time.perf_counter()
CHUNK, CONTEXT, DECODE = 32, 1024, 8
PROMPT = [1000 + i % 1000 for i in range(CHUNK)]


def save(model: str, out: str) -> None:
    from tinyllm import tg
    from tinyllm.tg.upstream import UpstreamSession

    session = UpstreamSession(model, CONTEXT, rows=1, chunk=CHUNK)
    ready = time.perf_counter()
    # warm both programs past JIT capture, then record the reference continuation from position 0
    session.prefill(PROMPT * 3)
    for _ in range(4):
        session.step()
    warm = time.perf_counter()
    session.reset()
    out_ids = session.prefill(PROMPT)
    tokens = [int(out_ids[0])]
    for _ in range(DECODE - 1):
        tokens.append(int(session.step()[0]))
    with open(out, "wb") as f:
        pickle.dump({"prefill": session.model.prefill_jit, "rollout": session.model.rollout_jit, "tokens": tokens}, f)
    size = __import__("os").path.getsize(out)
    print(f"device {tg.default_device()}")
    print(f"normal start: import+load {ready - PROCESS_START:.2f} s, warm-up {warm - ready:.2f} s")
    print(f"reference tokens {tokens}")
    print(f"pickled both programs: {size / 1e6:.0f} MB")


def load(path: str) -> None:
    import numpy as np

    from tinyllm import tg

    imported = time.perf_counter()
    with open(path, "rb") as f:
        saved = pickle.load(f)
    unpickled = time.perf_counter()

    v_pos = tg.UOp.variable("start_pos", 0, CONTEXT - 1)
    v_toks = tg.UOp.variable("toks", 1, CHUNK)
    temp = tg.Tensor([0.0])
    buf = np.zeros((1, CONTEXT), dtype=np.int32)
    buf[0, :CHUNK] = PROMPT
    table = tg.Tensor(buf)
    lo, count = v_pos.bind(0), v_toks.bind(CHUNK)
    out = saved["prefill"](table[:, lo : lo + count].contiguous(), temp, lo).realize()
    tokens = [int(out.tolist()[0][0])]
    first = time.perf_counter()
    for i in range(DECODE - 1):
        out = saved["rollout"](out.contiguous(), temp, v_pos.bind(CHUNK + i)).realize()
        tokens.append(int(out.tolist()[0][0]))
    done = time.perf_counter()
    print(f"device {tg.default_device()}")
    print(
        f"pickled start: import {imported - PROCESS_START:.2f} s, unpickle {unpickled - imported:.2f} s, "
        f"first token {first - unpickled:.2f} s, {DECODE - 1} more tokens {done - first:.2f} s"
    )
    print(f"total to first token {first - PROCESS_START:.2f} s")
    print(f"tokens {tokens}")
    print("MATCH" if tokens == saved["tokens"] else f"MISMATCH (expected {saved['tokens']})")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "save":
        save(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "load":
        load(sys.argv[2])
    else:
        sys.exit(__doc__)
