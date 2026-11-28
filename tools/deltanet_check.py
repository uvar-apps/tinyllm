#!/usr/bin/env python3
"""The DeltaNet block's fused prefill path (tinyllm.tg.kernels.deltafuse) against tinygrad's own.

    DEV=KFD+AMD uv run python tools/deltanet_check.py <model.gguf> [--tokens 256] [--blocks 2]

For a few of the model's DeltaNet blocks, from the same random input and the same random recurrent and convolution
state, both paths compute a prefill call at position 0 (where the state is reset) and at a later position. The
block's output, the recurrent state and the convolution state after the call must agree to rounding: the fused
kernels sum the convolution's taps and the norms in another order than tinygrad's generic kernels, nothing more.

Then the same for a padded call (`tinyllm.tg.speculative.padded`): rows past `real` are padding, and the state
after row `mid` - 1 is kept. The fused path is compared with the padded path on tinygrad's kernels and the scan in
tinygrad's layout, including the kept state.

Exits non-zero when a difference exceeds its bound.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

STATE, OUTPUT = 2e-4, 5e-3  # relative to the largest value: float32 states, an output that went through fp16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=256, help="rows of the call (a multiple of 256, at least 512 pads)")
    ap.add_argument("--blocks", type=int, default=2, help="DeltaNet blocks to check: the first, then spread out")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from tinyllm.tg import Tensor, UOp, default_device  # chooses the device, before anything imports tinygrad
    from tinyllm.tg import speculative as spec
    from tinyllm.tg.kernels import deltafuse, deltanet
    from tinyllm.tg.upstream import SMALL_CHUNK, GatedDeltaNetBlock, UpstreamSession

    s = UpstreamSession(Path(args.model).expanduser(), 4096, rows=1, chunk=2048)
    model, cfg = s.model, s.model.config
    blocks = [b for b in model.blk if isinstance(b, GatedDeltaNetBlock)]
    if not blocks:
        raise SystemExit("the model has no DeltaNet blocks")
    spec.recurrent_blocks(model)  # the padded path lives on tinyllm's block class
    picked = [blocks[i * (len(blocks) - 1) // max(args.blocks - 1, 1)] for i in range(min(args.blocks, len(blocks)))]
    T, rng = args.tokens, np.random.default_rng(0)
    variable = UOp.variable("start_pos", 0, int(model.max_context) - 1)
    failures = 0

    def check(name: str, got: np.ndarray, want: np.ndarray, bound: float) -> None:
        nonlocal failures
        scale = max(float(np.max(np.abs(want))), 1e-30)
        worst = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))) / scale
        ok = bool(np.isfinite(got).all()) and worst <= bound
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {worst:.3g} of the largest value (bound {bound:g})")

    for b in picked:
        index = model.blk.index(b)
        x = Tensor((rng.standard_normal((1, T, cfg.dim)) * 0.5).astype(np.float16)).contiguous().realize()
        b._init_state(x)
        b.allocate_mid(default_device())
        Tensor.realize(b.recurrent_state, b.conv_state)
        states: list[Any] = [b.recurrent_state, b.conv_state, b.mid_state, b.mid_conv]
        start = [
            (rng.standard_normal(tuple(int(n) for n in t.shape)) * scale).astype(np.float32)
            for t, scale in zip(states, (0.05, 1.0, 0.0, 0.0), strict=True)
        ]

        def run(
            fused: bool, position: int, padding: Any, b: Any = b, x: Any = x, start: Any = start, states: Any = states
        ) -> list[np.ndarray]:
            Tensor.realize(*(t.assign(Tensor(a)) for t, a in zip(states, start, strict=True)))
            lanes = deltanet.ROW_LANES
            deltafuse.ACTIVE, deltanet.ROW_LANES, b.padding = fused, lanes if fused else 0, padding
            try:
                out = b._attention(x, (variable.bind(position),), None).float().realize()
            finally:
                deltafuse.ACTIVE, deltanet.ROW_LANES, b.padding = True, lanes, None
            return [out.numpy(), *(t.numpy().copy() for t in states)]

        for position in (0, 1000):
            want, got = run(False, position, None), run(True, position, None)
            name = f"block {index}, {T} rows at position {position}"
            check(f"{name}: output", got[0], want[0], OUTPUT)
            check(f"{name}: recurrent state", got[1], want[1], STATE)
            check(f"{name}: convolution state", got[2], want[2], STATE)
        if T >= 2 * SMALL_CHUNK:
            mid, real = T - SMALL_CHUNK, T - 68
            padding = (UOp.variable(f"real_{T}", mid + 1, T).bind(real), mid)
            want, got = run(False, 1000, padding), run(True, 1000, padding)
            name = f"block {index}, {T} rows of which {real} real, state kept at {mid}"
            check(f"{name}: output of the real rows", got[0][:, :real], want[0][:, :real], OUTPUT)
            for label, i in (
                ("recurrent state", 1),
                ("convolution state", 2),
                ("kept recurrent state", 3),
                ("kept convolution state", 4),
            ):
                check(f"{name}: {label}", got[i], want[i], STATE)
    print(f"tinyllm deltanet check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
