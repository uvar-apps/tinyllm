#!/usr/bin/env python3
"""The DeltaNet prefill scan kernels (tinyllm.tg.kernels.deltanet.gated_delta_rows) against the recurrence itself.

    DEV=KFD+AMD uv run python tools/delta_scan_check.py [--tokens 24] [--spans 0,4,2,8,16]

No model: random operands at Qwen3.8's head sizes. For every layout of the scan (tinygrad's, a state row over
all 32 lanes, written 0 here; a row on 4 lanes, the default; and the other spans the kernel takes), with one key
head per value head and with key heads shared, from position 0 (where the state is reset) and from a later
one, the outputs and the state after the call must match the recurrence computed in float64:

    S <- alpha S;  d = beta (v - S k);  S <- S + d k^T;  out = S q

Then the same with the state after a row inside the call kept in another buffer, as a padded prefill call keeps it
(`tinyllm.tg.speculative.padded`): the kernels run stretch by stretch, the first one from the state's buffer into
the kept one.

Exits non-zero when a difference exceeds 1e-4 of the largest value (float32 sums in another order give 2e-7).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

BOUND = 1e-4
HEADS, KEY_HEADS, VALUE_DIM, KEY_DIM = 6, 2, 64, 128


def recurrence(ops: tuple[Any, ...], state: np.ndarray, reset: bool, keep: int) -> tuple[Any, Any, Any]:
    """Outputs, final state, and the state after row `keep` - 1 (None when `keep` is 0), in float64."""
    q, k, v, beta, alpha = ops
    S = np.zeros(state.shape) if reset else state.astype(np.float64)
    out, kept = np.zeros(v.shape), None
    for t in range(v.shape[1]):
        if keep and t == keep:
            kept = S.copy()
        for h in range(v.shape[0]):
            kh = h % q.shape[0]
            decayed = S[h] * alpha[h, t]
            delta = (v[h, t] - decayed @ k[kh, t]) * beta[h, t]
            S[h] = decayed + delta[:, None] * k[kh, t][None, :]
            out[h, t] = S[h] @ q[kh, t]
    return out, S, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--spans", default="0,4,2,8,16", help="lanes per state row; 0: tinygrad's layout")
    ap.add_argument("--quick", action="store_true", help="accepted for the test suite; the check is small as it is")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from tinyllm import device

    device.configure("amd")  # before anything imports tinygrad
    from tinyllm.tg import Tensor, UOp
    from tinyllm.tg.kernels import deltanet

    T, rng, failures = args.tokens, np.random.default_rng(0), 0
    spans = [int(s) for s in args.spans.split(",")]
    default = deltanet.ROW_LANES

    def worst(got: np.ndarray, want: np.ndarray) -> float:
        return float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))

    for shared in (False, True):
        key_heads = KEY_HEADS if shared else HEADS
        q = (rng.standard_normal((key_heads, T, KEY_DIM)) / KEY_DIM**0.5).astype(np.float32)
        k = (rng.standard_normal((key_heads, T, KEY_DIM)) / KEY_DIM**0.5).astype(np.float32)
        v = rng.standard_normal((HEADS, T, VALUE_DIM)).astype(np.float32)
        beta = rng.uniform(0.2, 0.9, (HEADS, T)).astype(np.float32)
        alpha = rng.uniform(0.8, 1.0, (HEADS, T, 1)).astype(np.float32)
        start = (rng.standard_normal((HEADS, VALUE_DIM, KEY_DIM)) * 0.1).astype(np.float32)
        dots = (q * k).sum(-1)
        ops = (q, k, v, beta, alpha[..., 0])
        for position, keep in ((0, 0), (7, 0), (7, T // 2), (0, T // 2)):
            want = recurrence(ops, start, position == 0, keep)
            for span in spans:
                deltanet.ROW_LANES = span
                try:
                    state = Tensor(start[None]).contiguous().realize()
                    kept = Tensor.zeros(*state.shape).contiguous().realize()
                    pos = Tensor(UOp.variable("start_pos", 0, 1000).bind(position))
                    tensors = [Tensor(a[None]).contiguous().realize() for a in (q, k, v, beta, alpha)]
                    kq = Tensor(dots[None]).contiguous().realize() if shared else None
                    snapshots = [(keep, kept.uop)] if keep else []
                    core = deltanet.gated_delta_rows(*tensors, state.uop, snapshots, pos, (), kq)
                    got = [core.realize().numpy()[0], state.numpy()[0], kept.numpy()[0]]
                finally:
                    deltanet.ROW_LANES = default
                errors = [worst(got[0], want[0]), worst(got[1], want[1])]
                errors += [worst(got[2], want[2])] if keep else []
                ok = all(np.isfinite(g).all() for g in got) and max(errors) <= BOUND
                failures += not ok
                layout = f"a row on {span} lanes" if span else "tinygrad's layout"
                heads = "shared key heads" if shared else "a key head per value head"
                where = f"position {position}" + (f", state kept at row {keep}" if keep else "")
                names = ("output", "state", "kept state")
                found = ", ".join(f"{n} {e:.2g}" for n, e in zip(names, errors, strict=False))
                print(f"{'ok  ' if ok else 'FAIL'} {layout}, {heads}, {where}: {found}")
    print(f"tinyllm delta scan check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
