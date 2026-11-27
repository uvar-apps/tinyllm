#!/usr/bin/env python3
"""A padded prefill call of Qwen3.8-Flash-Next against the calls it replaces.

    DEV=KFD+AMD TINYLLM_STATIC_SIZES=768 uv run python tools/padded_call_check.py <...-00001-of-00004.gguf>

Where the model drafts with its MTP block, a prompt's last stretch runs as one call of a fixed length, its rows past
the prompt's end padding, and the state for the next request's checkpoint is the one that call keeps 256 rows before
its end (`tinyllm.tg.upstream.UpstreamSession._prefill_padded`). This tool runs the same tokens from position 0
both ways and checks, for calls with no padding, some, and all but one row of the last 256:

1. The call ends in the state after its last real row: every DeltaNet block's recurrent and convolution state and
   the n-gram layer's convolution state are far closer to the state plain calls reach at that position than the
   state one row on is. They cannot be equal bit for bit: programs of other lengths round differently, as two
   ways of cutting the plain calls do. A wrong row would be as far as the next one.
2. The state it keeps inside is the state at that row, by the same measure, and `prefill` names that position.
3. Its logits pick the token the plain calls' logits pick, or one tied with it within their difference.
4. Nothing in the state is infinite or not a number: the padding rows compute on zeros, not on what was there.

It runs the padded program three times before changing the count of real rows, so the last runs replay the
captured program. It prints the greedy tokens decoded after a padded call and after plain calls for the record.
Exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("recurrent", "convolution", "n-gram convolution")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mtp", default=None, help="the MTP block's GGUF (default: found as `tinyllm serve` does)")
    ap.add_argument("--text", default=None, help="a text file for the prompt (default: this repository's sources)")
    ap.add_argument("--length", type=int, default=768, help="the fixed length to check (one of the session's)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from tinyllm.server.serve import find_mtp
    from tinyllm.tg import synchronize  # chooses the device, before anything imports tinygrad
    from tinyllm.tg.upstream import SMALL_CHUNK, UpstreamSession

    model_path = Path(args.model).expanduser()
    mtp = Path(args.mtp).expanduser() if args.mtp else find_mtp(model_path)
    if mtp is None:
        raise SystemExit(f"no MTP block found for {model_path.name}; pass --mtp")
    s = UpstreamSession(model_path, 8192, rows=1, chunk=2048, logits=True, speculative=4, mtp_path=mtp)
    length, inside = args.length, args.length - SMALL_CHUNK
    if not s.pads or length not in s._static or inside <= 0:
        raise SystemExit(f"the session has no padded program of {length} tokens (it has {sorted(s._static)})")
    if args.text:
        text = Path(args.text).read_text()
    else:  # code, as the model is used: this repository's own sources
        text = "\n".join(p.read_text() for p in sorted((Path(__file__).parent.parent / "src").rglob("*.py"))[:40])
    ids = s.tokenizer().encode(text)[: length + 1]
    if len(ids) <= length:
        raise SystemExit(f"the text has {len(ids)} tokens, {length + 1} are needed")
    some, one = length - 68, inside + 1  # real rows of the padded calls, besides all of them
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' if detail else ''}{detail}")

    def states(tensors: list[Any]) -> list[np.ndarray]:
        synchronize()
        return [t.numpy().copy() for t in tensors]

    def run(cuts: list[int], **last: Any) -> tuple[dict[int, list[np.ndarray]], dict[int, np.ndarray]]:
        """ids[: cuts[-1]] from position 0 in calls that end at `cuts`: the state and the logits after each."""
        s.reset()
        state, logits, lo = {}, {}, 0
        for hi in cuts:
            out = s.prefill(ids[lo:hi], wait_each=False, **(last if hi == cuts[-1] else {}))
            state[hi], logits[hi], lo = states(s._state_tensors()), np.asarray(out[0]).copy(), hi
        return state, logits

    def distance(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float]:
        worst = dict.fromkeys(KINDS, 0.0)
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            kind = KINDS[2] if i == len(a) - 1 else KINDS[i % 2]
            worst[kind] = max(worst[kind], float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
        return worst

    def close(name: str, got: list[np.ndarray], want: list[np.ndarray], next_row: list[np.ndarray]) -> None:
        near, row = distance(got, want), distance(next_row, want)
        share = {k: near[k] / max(row[k], 1e-30) for k in KINDS}
        detail = ", ".join(f"{k} {near[k]:.3g} ({share[k]:.2g})" for k in KINDS)
        check(f"{name}, as a share of one row's change", max(share.values()) <= 0.5, detail)

    def same_choice(name: str, got: np.ndarray, want: np.ndarray) -> None:
        worst = float(np.abs(got - want).max())
        tie = float(want.max() - want[int(got.argmax())]) <= worst
        check(
            f"{name}: the same top token, or a tie",
            int(got.argmax()) == int(want.argmax()) or tie,
            f"max diff {worst:.3g}",
        )

    print(f"plain calls ending at {inside}, {one}, {some}, {some + 1}, {length - 1} and {length}")
    ref, ref_logits = run([inside, one, some, some + 1, length - 1, length])
    again, _ = run([inside, some])
    near, row = distance(again[some], ref[some]), distance(ref[some + 1], ref[some])
    cut = ", ".join(f"{k} {near[k] / max(row[k], 1e-30):.2g}" for k in KINDS)
    print(f"info the plain calls cut another way, at {some}, as a share of one row's change: {cut}")

    for real, neighbour, label in (
        (length, length - 1, "first run"),
        (length, length - 1, "capturing run"),
        (length, length - 1, "captured"),
        (some, some + 1, "captured"),
        (one, inside, "captured"),
    ):
        seen: list[int] = []
        got, logits = run([real], ends=True, inside_last=seen.append)
        name = f"{length}-token call with {real} real rows ({label})"
        placed = s.pos == real and s.padded_end == length and seen == [inside]
        check(
            f"{name}: ends at {real}, padding to {length}, checkpoint at {inside}",
            placed,
            f"{s.pos}, {s.padded_end}, {seen}",
        )
        close(f"{name}: the state it ends in", got[real], ref[real], ref[neighbour])
        close(f"{name}: the state it keeps at {inside}", states(s._mid_tensors()), ref[inside], ref[one])
        same_choice(f"{name}: logits", logits[real], ref_logits[real])
        check(f"{name}: every state is finite", all(bool(np.isfinite(x).all()) for x in got[real]))

    decoded = {}
    for name, cuts, last in (("plain calls", [inside, some], {}), ("a padded call", [some], {"ends": True})):
        _, logits = run(cuts, **last)
        out = [int(logits[some].argmax())]
        for _ in range(12):
            out.append(int(np.asarray(s.step([out[-1]])[0]).argmax()))
        decoded[name] = out
    agree = sum(a == b for a, b in zip(*decoded.values(), strict=True))
    print(f"info greedy tokens after {some}: {agree} of 13 the same after a padded call as after plain calls")
    print(f"tinyllm padded call check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
