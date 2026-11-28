#!/usr/bin/env python3
"""The expert selection kernel (tinyllm.tg.kernels.moe_select) and the routing tables built in chunks
(tinyllm.tg.kernels.moe_mm.route), against NumPy and against the paths they replace.

    DEV=KFD+AMD uv run python tools/moe_route_check.py [--rows 2048] [--time]

No model. Selection: random router logits at Qwen3.8-Flash-Next's sizes (512 experts, 10 a token) and at others,
with equal values, signed zeros and a wide range mixed in. For every row:

- the selected experts must be exactly `pairwise_topk`'s (tinygrad.llm.model): the k largest logits, ascending,
  equal values by lower index first, computed here in NumPy and by tinygrad on the device;
- the weights must match the softmax over the selected logits, computed in float64, to 1e-6, and tinygrad's own
  bit for bit (with the scaling factor applied after, as the model applies it).

A further case fills rows with NaN, infinities and random bits, as the rows of padding of a padded call may hold:
the kernel must still select k distinct experts in range.

Routing: for random selections, even and skewed, with and without padding pairs, the tables built by a workgroup
per chunk of pairs must be those of the one-workgroup kernel, entry for entry, and the pairs sorted by expert must
be NumPy's stable sort.

Exits non-zero on any difference. `--time` prints the time per call of each path and the one it replaces.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

BOUND = 1e-6


def reference(logits: np.ndarray, k: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """`pairwise_topk`'s selection and the softmax of the selected logits, in float64."""
    rows, n = logits.shape
    index = np.broadcast_to(np.arange(n), (rows, n))
    order = np.lexsort((-index, logits + 0.0), axis=-1)  # ascending by value; equal values: the higher index first
    sel = order[:, n - k :]
    picked = np.take_along_axis(logits.astype(np.float64), sel, axis=-1)
    e = np.exp(picked - picked.max(-1, keepdims=True))
    return sel.astype(np.int32), e / e.sum(-1, keepdims=True) * scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--time", action="store_true", help="time both paths at Flash-Next's sizes")
    ap.add_argument("--quick", action="store_true", help="accepted for the test suite; the check is small as it is")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from tinyllm import device

    device.configure("amd")  # before anything imports tinygrad
    from tinygrad.llm.model import pairwise_topk

    from tinyllm import tg
    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import moe_mm, moe_select

    rng, failures = np.random.default_rng(0), 0

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")

    for rows, n, k, scale in ((args.rows, 512, 10, 2.5), (256, 512, 8, 1.0), (128, 256, 4, 1.0), (384, 1024, 16, 1.0)):
        x = (rng.standard_normal((rows, n)) * 3).astype(np.float32)
        x[: rows // 4] = np.round(x[: rows // 4])  # many equal values
        x[rows // 4 : rows // 2] *= rng.choice(np.float32([1e-30, 1.0, 1e4]), (rows // 4, 1))
        x[::7, ::5] = np.float32(0.0)
        x[::7, 1::5] = np.float32(-0.0)
        x[3] = np.float32(1.25)  # one value everywhere: the first k experts
        logits = Tensor(x[None]).contiguous().realize()
        name = f"{rows} rows, {n} experts, {k} a token"
        if not moe_select.serves(logits, k):
            report(name, False, "not a call the kernel takes")
            continue
        sel, probs = moe_select.select(logits, k)
        got_sel, got_probs = sel.realize().numpy()[0], (probs * scale).realize().numpy()[0]
        want_sel, want_probs = reference(x, k, scale)
        wrong = int((got_sel != want_sel).any(-1).sum())
        report(f"{name}: experts against NumPy", wrong == 0, f"{wrong} row(s) differ")
        diff = float(np.abs(got_probs - want_probs).max()) / scale
        report(f"{name}: weights against NumPy", bool(np.isfinite(got_probs).all()) and diff <= BOUND, f"{diff:.2g}")
        _, own_sel = pairwise_topk(logits, k)
        own_probs = (logits.gather(-1, own_sel).softmax(-1) * scale).realize().numpy()[0]
        wrong = int((got_sel != own_sel.realize().numpy()[0]).any(-1).sum())
        report(f"{name}: experts against tinygrad", wrong == 0, f"{wrong} row(s) differ")
        diff = float(np.abs(got_probs - own_probs).max()) / scale
        report(f"{name}: weights against tinygrad", diff == 0.0, f"{diff:.2g} (the same bits)")

    rows, n, k = 256, 512, 10
    bits = rng.integers(0, 2**32, (rows, n), dtype=np.uint64).astype(np.uint32)
    junk = bits.view(np.float32).copy()
    junk[::3] = np.float32(np.nan)
    junk[1::9, ::2] = np.float32(np.inf)
    junk[2::9, ::2] = np.float32(-np.inf)
    sel, _ = moe_select.select(Tensor(junk[None]).contiguous().realize(), k)
    got_sel = sel.realize().numpy()[0]
    in_range = bool(((got_sel >= 0) & (got_sel < n)).all())
    distinct = all(len(set(row)) == k for row in got_sel.tolist())
    detail = f"in range {in_range}, distinct {distinct}"
    report("rows of NaN, infinities and random bits", in_range and distinct, detail)

    # the routing tables: a workgroup per chunk of pairs against the one-workgroup kernel
    def tables_of(sel_np: np.ndarray, experts: int, tokens: int, chunk: int) -> tuple[np.ndarray, list[np.ndarray]]:
        saved = moe_mm.ROUTE_CHUNK
        moe_mm.ROUTE_CHUNK = chunk
        try:
            classes, order = moe_mm.route(Tensor(sel_np).contiguous().realize(), experts, tokens)
            found = [order.realize().numpy()[: int((sel_np >= 0).sum())]]
            for _, (expert, first, count, total) in classes:
                n = int(total.realize().numpy()[0])
                found += [np.array([n]), *(t.realize().numpy()[:n] for t in (expert, first, count))]
        finally:
            moe_mm.ROUTE_CHUNK = saved
        return found[0], found[1:]

    for tokens, k, experts, kind in (
        (args.rows, 10, 512, "even"),
        (args.rows, 10, 512, "skewed"),
        (300, 10, 512, "skewed"),
        (1536, 10, 512, "padded"),
        (256, 8, 64, "even"),
    ):
        weights = 1.0 / (1 + np.arange(experts)) ** (1.5 if kind == "skewed" else 0.0)
        picks = [rng.choice(experts, k, replace=False, p=weights / weights.sum()) for _ in range(tokens)]
        sel_np = np.array(picks, dtype=np.int32)
        if kind == "padded":  # the rows of padding of a padded call: no expert
            sel_np[tokens - 200 :] = -1
        sel_np = sel_np.reshape(-1)
        name = f"routing, {tokens} tokens, {k} of {experts} experts, {kind}"
        if sel_np.size < moe_mm.ROUTE_MIN_PAIRS:
            report(name, False, "too few pairs for chunks")
            continue
        order, tables = tables_of(sel_np, experts, tokens, moe_mm.ROUTE_CHUNK)
        one_order, one_tables = tables_of(sel_np, experts, tokens, 0)
        valid = np.flatnonzero(sel_np >= 0)
        stable = valid[np.argsort(sel_np[valid], kind="stable")]
        pairs = zip(tables, one_tables, strict=False)
        same = len(tables) == len(one_tables) and all(np.array_equal(a, b) for a, b in pairs)
        report(f"{name}: tables against one workgroup", same and np.array_equal(order, one_order), f"{same}")
        report(f"{name}: pairs by expert against NumPy", np.array_equal(order, stable), f"{len(stable)} pairs")

    if args.time:
        tg.set_polling_wait()
        sel_t = Tensor(rng.integers(0, 512, args.rows * 10, dtype=np.int32)).contiguous().realize()

        def routing(t: Tensor) -> Tensor:
            classes, order = moe_mm.route(t, 512, args.rows)
            return Tensor.realize(order, *(u for _, group in classes for u in group))[0]

        for label, chunk in (("routing in chunks", moe_mm.ROUTE_CHUNK), ("routing in one workgroup", 0)):
            saved, moe_mm.ROUTE_CHUNK = moe_mm.ROUTE_CHUNK, chunk
            jit, times = tg.TinyJit(routing), []
            for i in range(43):
                start = time.perf_counter()
                jit(sel_t)
                tg.synchronize()
                if i >= 3:
                    times.append(time.perf_counter() - start)
            moe_mm.ROUTE_CHUNK = saved
            print(f"{label}: {1e3 * float(np.median(times)):.3f} ms a call at {args.rows} rows")
        rows, n, k = args.rows, 512, 10
        logits = Tensor((rng.standard_normal((1, rows, n)) * 3).astype(np.float32)).contiguous().realize()

        def ours(t: Tensor) -> Tensor:
            sel, probs = moe_select.select(t, k)
            return (sel.float() + probs * 2.5).realize()

        def own(t: Tensor) -> Tensor:
            _, sel = pairwise_topk(t, k)
            return (sel.float() + t.gather(-1, sel).softmax(-1) * 2.5).realize()

        for label, fxn in (("the kernel", ours), ("tinygrad's path", own)):
            jit, times = tg.TinyJit(fxn), []
            for i in range(43):
                start = time.perf_counter()
                jit(logits)
                tg.synchronize()
                if i >= 3:
                    times.append(time.perf_counter() - start)
            print(f"{label}: {1e3 * float(np.median(times)):.3f} ms a call at {rows} rows (with one generic kernel)")

    print(f"tinyllm moe route check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
