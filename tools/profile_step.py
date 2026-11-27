#!/usr/bin/env python3
"""Per-kernel time of one warm model call on the GPU.

    uv run python tools/profile_step.py <model.gguf> [--rows 1] [--kind decode|logits|sampled|prefill] [--top 20]
    uv run python tools/profile_step.py <model.gguf> --kind prefill --set q45k.GROUPS_PER_STEP=4  # override
    uv run python tools/profile_step.py <model.gguf> --kind prefill --chunk 2048 --context 2048 --text code.txt

Runs the model in a child process with tinygrad's DEBUG=2, which times every kernel on the device.
The model is warmed past JIT capture, then one call is marked and its kernels are grouped by name.
Times include the synchronisation DEBUG=2 adds after each kernel, so the total is larger than a
normal step; use the shares to find where time goes and `tinyllm bench` for absolute speed.

Kernel names: `linear_<type>` and `moe_<type>` are tinygrad's packed-weight kernels, `*_f16_wmma`
their matrix-core prefill variants, `linear_f16_gemv` the small dense kernel, `flash_*` and
`gated_delta_prefill` attention and the recurrent scan. Names starting with `r_` (reductions) or
`E_` (elementwise) are tinygrad's generic code generation: a large share there means tensors that
missed the fast kernels.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

MARK_START, MARK_END = "=== TINYLLM PROFILE START", "=== TINYLLM PROFILE END"
_LINE = re.compile(r"^\*\*\* \S+\s+\d+ (?P<name>.+?)\s+arg\s+\d+ mem\s+[\d.]+ GB tm\s+(?P<t>[\d.]+)(?P<unit>us|ms|s)/")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0}

CHILD = r"""
import ast, importlib, sys
from tinyllm import device
device.configure("amd")
from tinyllm.tg.upstream import UpstreamSession
path, rows, kind, chunk, context = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
max_context = int(sys.argv[6]) or context + 2048
for setting in sys.argv[7:]:  # module.NAME=value, applied to tinyllm.tg.kernels.<module>
    target, value = setting.split("=", 1)
    module, name = target.rsplit(".", 1)
    setattr(importlib.import_module(f"tinyllm.tg.kernels.{module}"), name, ast.literal_eval(value))
import os
k = int(kind[len("speculate"):] or 3) if kind.startswith("speculate") else 0
drafter = os.environ.get("TINYLLM_PROFILE_DRAFTER") or None  # a DFlash2 GGUF; the MTP block otherwise
logits = bool(k) or kind in ("logits", "sampled")
s = UpstreamSession(path, max_context, rows=rows, chunk=chunk, logits=logits, speculative=k, drafter=drafter)
greedy = [0.0, 32, 1.0, 0.0, 0.0]  # the GPU sampler's settings: temperature, top_k, top_p, min_p, u
def one():
    if kind == "logits": return s.step([1000] * rows)
    if kind == "sampled": return s.decode_sampled([1000], greedy)
    return s.step()
text = os.environ.get("TINYLLM_PROFILE_TEXT")  # a text file: its tokens instead of the synthetic ids
call = int(os.environ.get("TINYLLM_PROFILE_CALL") or chunk)  # tokens of the timed prefill call
if text:
    ids = s.tokenizer().encode(open(text).read())
    assert len(ids) >= context + call, f"{text} has {len(ids)} tokens, fewer than --context + the call"
    fill, prompt = ids[:context], ids[context:context + call]
else:
    fill, prompt = [1000 + i % 1000 for i in range(context)], [1000 + i % 1000 for i in range(call)]
for _ in range(4): s.prefill(prompt)
if text: s.reset()  # the fill and the timed call then see the text in order
if context > s.pos:  # fill the context first, without printing a timing line for each of its kernels
    shown, sys.stdout = sys.stdout, open(os.devnull, "w")
    s.prefill(fill[s.pos:context], wait_each=False)
    sys.stdout = shown
if k:
    rows_ = [[0.0, 1.0, 1.0, 0.0, 0.0]] * (k + 1)
    for _ in range(4): s.speculate(1000, rows_)
else:
    for _ in range(4): one()
print("MARK_START", flush=True)
s.prefill(prompt) if kind == "prefill" else s.speculate(1000, rows_) if k else one()
print("MARK_END", flush=True)
""".replace("MARK_START", MARK_START).replace("MARK_END", MARK_END)


def parse(text: str) -> list[tuple[str, float]]:
    """(kernel name, seconds) for each kernel between the markers."""
    inside, kernels = False, []
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).rstrip()
        if line == MARK_START:
            inside = True
        elif line == MARK_END:
            break
        elif inside and (m := _LINE.match(line)):
            kernels.append((m["name"].strip(), float(m["t"]) * _SCALE[m["unit"]]))
    return kernels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument(
        "--kind",
        default="decode",
        help="decode (sample mode), logits (a decode returning the logits), sampled (a decode with the GPU sampler, "
        "as the server runs), prefill, or speculate[K]: one draft-and-verify cycle of K drafts (default 3), split "
        "into the drafting part and the verify part at the first DeltaNet scan of the verify",
    )
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--context", type=int, default=0, help="prefill this many tokens first: profile at that position")
    ap.add_argument(
        "--max-context",
        type=int,
        default=0,
        help="the session's context size (default: --context + 2048); Qwen3.8-Flash-Next's sparse attention "
        "indexer runs only above 2,048, and its cost follows this size, not the position",
    )
    ap.add_argument("--drafter", default=None, help="with --kind speculate: a DFlash2 GGUF instead of the MTP block")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="MODULE.NAME=VALUE",
        help="override a kernel module setting, e.g. q45k.GROUPS_PER_STEP=4 (repeatable)",
    )
    ap.add_argument("--source", metavar="PATH", help="write the child's output with every kernel's source here")
    ap.add_argument(
        "--text",
        metavar="PATH",
        help="prefill this file's tokens (--context of them first) instead of synthetic ids; Qwen3.8-Flash-Next's "
        "block selection and expert routing follow the content, so its profiles need real text",
    )
    ap.add_argument(
        "--call",
        type=int,
        default=0,
        help="with --kind prefill: tokens of the timed call (default --chunk); fewer than --chunk take the "
        "session's partial-chunk or short-call program, as a short prompt does in the server",
    )
    args = ap.parse_args()

    from tinyllm.weights.resolve import find_model

    model = str(find_model(args.model))
    env = {**os.environ, "DEBUG": "2"}
    if args.source:  # DEBUG=4 also prints every kernel's source as it is compiled
        env["DEBUG"] = "4"
    if args.drafter:
        env["TINYLLM_PROFILE_DRAFTER"] = os.path.expanduser(args.drafter)
    if args.text:
        env["TINYLLM_PROFILE_TEXT"] = os.path.abspath(os.path.expanduser(args.text))
    if args.call:
        env["TINYLLM_PROFILE_CALL"] = str(args.call)
    cmd = [sys.executable, "-c", CHILD, model, str(args.rows), args.kind, str(args.chunk), str(args.context)]
    cmd += [str(args.max_context), *args.set]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if args.source:
        with open(args.source, "w") as f:
            f.write(proc.stdout + proc.stderr)
    kernels = parse(proc.stdout + proc.stderr)
    if not kernels:
        sys.exit(f"no kernel timings found (exit {proc.returncode}):\n" + "\n".join(proc.stderr.splitlines()[-15:]))

    by_name: dict[str, list[float]] = defaultdict(list)
    for name, t in kernels:
        by_name[name].append(t)
    total = sum(t for _, t in kernels)
    generic = sum(t for name, t in kernels if name.startswith(("r_", "E_")))
    tokens = f", {args.call or args.chunk} tokens" if args.kind == "prefill" else ""
    what = f"{args.kind}, {args.rows} row(s){tokens}"
    print(f"{os.path.basename(args.model)}: one {what}")
    share = 100 * generic / total
    print(f"{len(kernels)} kernels, {total * 1e3:.1f} ms summed; generic code generation (r_*, E_*) {share:.1f}%\n")
    print(f"{'ms':>8} {'share':>6} {'calls':>5}  kernel")
    for name, times in sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[: args.top]:
        print(f"{sum(times) * 1e3:8.2f} {100 * sum(times) / total:5.1f}% {len(times):5d}  {name}")
    if args.kind.startswith("speculate"):
        first = next((i for i, (name, _) in enumerate(kernels) if "gated_delta_row" in name), len(kernels))
        # the verify's first layer starts a few kernels before its scan: its projections and convolution
        start = max(0, first - 8)
        draft, verify = kernels[:start], kernels[start:]
        print(
            f"\ndrafting: {sum(t for _, t in draft) * 1e3:.1f} ms in {len(draft)} kernels; "
            f"verify: {sum(t for _, t in verify) * 1e3:.1f} ms in {len(verify)} kernels"
        )
        for title, part in (("drafting", draft), ("verify", verify)):
            groups: dict[str, float] = defaultdict(float)
            for name, t in part:
                groups[name] += t
            top = sorted(groups.items(), key=lambda kv: -kv[1])[:8]
            print(f"  {title}: " + ", ".join(f"{n} {t * 1e3:.1f}" for n, t in top))


if __name__ == "__main__":
    main()
