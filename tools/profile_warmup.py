#!/usr/bin/env python3
"""Where does start-up time go?

Profiles the calls tinygrad's JIT makes before it starts replaying compiled programs: the
first call (schedule, generate code, compile, run), the second (schedule again and capture),
and the third (first replay). Reports wall time per call and, from cProfile, the share of
Python time spent in each part of tinygrad, plus time spent inside the kernel compiler.

    uv run python tools/profile_warmup.py <model.gguf> [--device amd] [--ctx 4096]

cProfile slows Python-heavy code, so read the shares, and take absolute start-up times from
`tinyllm bench`.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from collections import defaultdict

BUCKETS = (
    ("tinygrad/llm/", "model graph construction (tinygrad.llm)"),
    ("tinygrad/engine/", "scheduling and JIT (tinygrad.engine)"),
    ("tinygrad/uop/", "graph rewriting (tinygrad.uop)"),
    ("tinygrad/codegen/", "kernel code generation (tinygrad.codegen)"),
    ("tinygrad/renderer/", "rendering source (tinygrad.renderer)"),
    ("tinygrad/runtime/", "device runtime (tinygrad.runtime)"),
    ("tinygrad/", "other tinygrad (tensor, helpers, dtype)"),
    ("~", "built-in and C calls (compiler library, ioctl waits, sqlite, hashing)"),
)


def bucket_of(filename: str) -> str:
    for needle, label in BUCKETS:
        if needle in filename:
            return label
    return "outside tinygrad"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    from tinyllm import device

    device.configure(args.device)
    from tinyllm import tg
    from tinyllm.tg.upstream import UpstreamSession

    # `compile` runs only when tinygrad's on-disk cache of compiled kernels misses
    compile_stats = {"calls": 0, "seconds": 0.0}
    compiler_type = type(tg.Device[tg.Device.DEFAULT].compiler)
    original = compiler_type.compile

    def timed_compile(self, src):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        try:
            return original(self, src)
        finally:
            compile_stats["calls"] += 1
            compile_stats["seconds"] += time.perf_counter() - start

    compiler_type.compile = timed_compile

    session = UpstreamSession(args.model, args.ctx, rows=1, chunk=args.chunk)
    print(f"device {tg.default_device()}, load {session.load_seconds:.1f} s")
    prompt = [1000 + i % 1000 for i in range(args.chunk)]

    calls = [("prefill call 1 (schedule, generate, compile, run)", lambda: session.prefill(prompt))]
    calls.append(("prefill call 2 (schedule again, JIT capture)", lambda: session.prefill(prompt)))
    calls.append(("prefill call 3 (first replay)", lambda: session.prefill(prompt)))
    calls.append(("decode call 1 (schedule, generate, compile, run)", lambda: session.step()))
    calls.append(("decode call 2 (schedule again, JIT capture)", lambda: session.step()))
    calls.append(("decode call 3 (first replay)", lambda: session.step()))

    total_by_bucket: dict[str, float] = defaultdict(float)
    total_wall = 0.0
    for label, fn in calls:
        before = dict(compile_stats)
        prof = cProfile.Profile()
        start = time.perf_counter()
        prof.enable()
        fn()
        prof.disable()
        wall = time.perf_counter() - start
        total_wall += wall
        by_bucket: dict[str, float] = defaultdict(float)
        for (filename, _line, _name), (_cc, _nc, tottime, _ct, _callers) in pstats.Stats(prof).stats.items():  # type: ignore[attr-defined]
            by_bucket[bucket_of(filename)] += tottime
        compiled = compile_stats["calls"] - before["calls"]
        compile_s = compile_stats["seconds"] - before["seconds"]
        print(f"\n{label}: {wall:.2f} s  ({compiled} compiler runs after cache misses, {compile_s:.2f} s)")
        for name, seconds in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
            if seconds >= 0.01 * wall:
                print(f"  {seconds:7.2f} s  {100 * seconds / wall:5.1f}%  {name}")
            total_by_bucket[name] += seconds

    print(f"\nall six calls: {total_wall:.1f} s under the profiler")
    for name, seconds in sorted(total_by_bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {seconds:7.2f} s  {100 * seconds / total_wall:5.1f}%  {name}")
    print(f"  compiler runs: {compile_stats['calls']}, {compile_stats['seconds']:.2f} s (part of the C-call bucket)")


if __name__ == "__main__":
    main()
