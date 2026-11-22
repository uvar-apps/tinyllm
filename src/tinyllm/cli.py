"""Command line entry point.

Subcommands import what they need lazily: `inspect` and `device` never import tinygrad,
and commands that run a model fix the device before tinygrad is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from tinyllm import __version__


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _cmd_inspect(args: argparse.Namespace) -> int:
    from tinyllm import inspection
    from tinyllm.weights.resolve import find_model

    store = None if args.store_gb is None else int(args.store_gb * 1e9)
    r = inspection.report(find_model(args.model), args.ctx, args.slots, store)
    print(json.dumps(r, indent=2) if args.json else inspection.render(r))
    return 0 if (r["plan"] is None or r["plan"]["fits"] is not False) else 2


def _cmd_device(args: argparse.Namespace) -> int:
    from tinyllm import device

    gpus = device.detect_amd_gpus()
    for g in gpus:
        status = "supported" if g.supported else "not supported by tinygrad's AMD backend"
        print(f"AMD GPU node {g.node}: {g.name or 'unnamed'}, {g.arch}, {g.compute_units} compute units ({status})")
    if not gpus:
        print("no AMD GPU exposed through the kernel driver")
    print(f"DEV={device.resolve(args.device)}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from tinyllm.bench import run

    return run.main(args)


def _cmd_oracle(args: argparse.Namespace) -> int:
    from tinyllm.oracle import commands

    return commands.main(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tinyllm", description="LLM serving on tinygrad for quantized GGUF models")
    p.add_argument("--version", action="version", version=f"tinyllm {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def device_arg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--device", default=None, help="auto, amd, cpu, nv, cuda or a tinygrad DEV string (default: auto)"
        )

    sp = sub.add_parser("inspect", help="describe a GGUF file and its memory plan without opening a device")
    sp.add_argument("model", help="GGUF path, or a file name under the model directory")
    sp.add_argument("--ctx", type=int, default=32768, help="context tokens per slot (default: %(default)s)")
    sp.add_argument("--slots", type=int, default=4, help="concurrent sequences (default: %(default)s)")
    sp.add_argument(
        "--store-gb", type=float, default=None, help="state store budget (default: one slot's worth per slot)"
    )
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(fn=_cmd_inspect)

    sp = sub.add_parser("device", help="show detected GPUs and the device tinyllm would use")
    sp.add_argument("--device", default="auto")
    sp.set_defaults(fn=_cmd_device)

    sp = sub.add_parser("bench", help="measure prefill and decode speed, optionally against a baseline")
    sp.add_argument("model")
    device_arg(sp)
    sp.add_argument("--rows", type=_ints, default=[1], help="rows per step, comma-separated (default: 1)")
    sp.add_argument("--chunk", type=_ints, default=[32], help="prefill tokens per call, comma-separated (default: 32)")
    sp.add_argument("--mode", type=_strs, default=["sample"], help="sample, logits or both (default: sample)")
    sp.add_argument("--prefill", type=int, default=256, help="prompt tokens per row (default: %(default)s)")
    sp.add_argument("--decode", type=int, default=32, help="decode steps (default: %(default)s)")
    sp.add_argument("--ctx", type=int, default=4096, help="context size (default: %(default)s)")
    sp.add_argument("--json", dest="json_out", metavar="PATH", help="write the full record here")
    sp.add_argument("--append", action="store_true", help="add these cases to an existing --json record")
    sp.add_argument("--compare", metavar="PATH", help="baseline record to compare against")
    sp.add_argument("--tolerance", type=float, default=0.03, help="allowed slowdown (default: %(default)s)")
    sp.set_defaults(fn=_cmd_bench)

    sp = sub.add_parser("oracle", help="capture and check reference outputs")
    osub = sp.add_subparsers(dest="oracle_command", required=True)
    op = osub.add_parser("capture", help="run tinygrad's own engine greedily and record tokens and top logits")
    op.add_argument("model")
    op.add_argument("--out", required=True, metavar="PATH")
    device_arg(op)
    op.add_argument("--prompts", metavar="PATH", help="JSON list of {id, text} (default: built-in set)")
    op.add_argument("--decode", type=int, default=16, help="tokens to generate per prompt (default: %(default)s)")
    op.add_argument("--ctx", type=int, default=1024)
    op.add_argument("--topk", type=int, default=8)
    op = osub.add_parser("check", help="recapture and compare with a stored record")
    op.add_argument("golden")
    op.add_argument("--model", help="model path (default: located by the file name stored in the record)")
    device_arg(op)
    op.add_argument("--atol", type=float, default=None, help="allowed logit difference (default: by device)")
    op = osub.add_parser("reference", help="compare a stored record with llama.cpp on the CPU")
    op.add_argument("golden")
    op.add_argument("--model")
    op.add_argument("--server", help="llama-server binary (default: TINYLLM_REF_LLAMA_SERVER or .ref/llama.cpp)")
    op.add_argument("--threads", type=int, default=16)
    sp.set_defaults(fn=_cmd_oracle)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"tinyllm: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
