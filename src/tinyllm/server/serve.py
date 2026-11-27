"""`tinyllm serve`: load a model, start the HTTP server on a thread, run the engine on the main thread."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

from tinyllm.runner.checkpoints import CHECKPOINTS

log = logging.getLogger("tinyllm.serve")

# GGUF keys of the model's recommended sampling settings -> SamplingParams names
SAMPLING_KEYS = {
    "general.sampling.temp": "temperature",
    "general.sampling.top_k": "top_k",
    "general.sampling.top_p": "top_p",
}


def sampling_defaults(kv: dict[str, Any]) -> dict[str, float]:
    return {name: float(kv[key]) for key, name in SAMPLING_KEYS.items() if key in kv}


DEFAULT_DRAFTS = 3  # MTP drafts per speculative cycle (docs/01-research-findings.md, section 5.7)
DFLASH_DRAFTS = 7  # DFlash2 drafts per cycle: its whole block


def has_mtp(path: Path, arch: str | None = None) -> bool:
    """Whether the GGUF carries one MTP (nextn) block, which speculative decoding drafts with (and, with `arch`,
    is of that architecture)."""
    from tinyllm.weights import gguf

    kv = gguf.read(path, header_only=True).kv
    if arch is not None and kv.get("general.architecture") != arch:
        return False
    return int(kv.get(f"{kv.get('general.architecture')}.nextn_predict_layers", 0) or 0) == 1


MTP_ORDER = ("shared-Q8_0", "shared-Q4_K_M", "shared", "")  # which of several MTP files to take, by name


def find_mtp(path: Path) -> Path | None:
    """The GGUF with the model's MTP block when it ships apart from the model (Qwen3.8-Flash-Next): an `mtp-*.gguf`
    of the model's architecture in the model's directory, the one above it, or an `MTP` directory of either.
    Files that share the model's embedding and head come first (`MTP_ORDER`)."""
    arch = architecture(path)
    folders = [d / sub for d in (path.parent, path.parent.parent) for sub in ("", "MTP")]
    found = sorted({f for d in folders if d.is_dir() for f in d.glob("mtp-*.gguf") if has_mtp(f, arch)})
    return min(found, key=lambda f: next(i for i, part in enumerate(MTP_ORDER) if part in f.name), default=None)


# Qwen3.8-Flash-Next (qwen4exp) prefills in passes of 2,048 tokens, which its grouped expert kernels need to read
# each expert's weights once for many rows, and lets tinygrad tune its generic kernels (BEAM search, cached on disk
# after the first start) (docs/04-qwen3.8-flash-next.md, section 14)
ARCH_CHUNK, ARCH_BEAM = {"qwen4exp": 2048}, {"qwen4exp": 2}
# (MTP, DFlash2) drafts per cycle where the defaults are not the fastest. Qwen3.8-Flash-Next: every verify row reads
# its own experts, so a long block costs more than it brings; 4 is the fastest on code and stays above plain
# decoding on prose (docs/04-qwen3.8-flash-next.md, section 15)
ARCH_DRAFTS = {"qwen4exp": (4, 4)}
# drafts of a block copied from earlier text, for models whose session verifies such blocks (0 elsewhere)
ARCH_COPY = {"qwen4exp": 15}
DEFAULT_CHUNK = 128


def architecture(path: Path) -> str:
    from tinyllm.weights import gguf

    return str(gguf.read(path, header_only=True).kv.get("general.architecture", ""))


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    import os

    from tinyllm.weights.resolve import find_model

    path = find_model(args.model)
    arch = architecture(path)
    chunk = args.chunk or ARCH_CHUNK.get(arch, DEFAULT_CHUNK)
    beam = args.beam if args.beam is not None else ARCH_BEAM.get(arch, 0)
    if beam and "JITBEAM" not in os.environ:  # read by tinygrad when it compiles the programs
        os.environ["JITBEAM"] = str(beam)
    from tinyllm import device

    device.configure(args.device)
    import uvicorn

    from tinyllm.engine.core import Engine
    from tinyllm.frontend.template import ChatTemplate
    from tinyllm.runner.upstream import UpstreamRunner
    from tinyllm.server.app import ServerConfig, create_app
    from tinyllm.server.client import EngineClient
    from tinyllm.tg import programs as tl_programs
    from tinyllm.tg.tokenizer import GGUFTokenizer

    start = time.perf_counter()
    drafter = Path(args.draft_model).expanduser() if args.draft_model else None
    mtp = None  # the MTP block's own GGUF, for a model that does not carry it
    if not drafter and not has_mtp(path) and args.speculative != 0:
        mtp = Path(args.mtp).expanduser() if args.mtp else find_mtp(path)
        if mtp is not None and not has_mtp(mtp, arch):
            raise SystemExit(f"{mtp} has no MTP block for {path.name}")
    mtp_drafts, dflash_drafts = ARCH_DRAFTS.get(arch, (DEFAULT_DRAFTS, DFLASH_DRAFTS))
    default = (dflash_drafts if drafter else mtp_drafts) if drafter or mtp or has_mtp(path) else 0
    k = args.speculative if args.speculative is not None else default
    copy = args.copy if args.copy is not None else ARCH_COPY.get(arch, 0)
    runner = UpstreamRunner(
        path,
        args.ctx,
        chunk=chunk,
        speculative=k,
        drafter=drafter if k else None,
        checkpoints=args.state_checkpoints,
        mtp_path=mtp if k else None,
        copy=copy if k and mtp else 0,
        programs=None if args.program_cache == "0" else args.program_cache or tl_programs.directory(),
    )
    tokenizer = GGUFTokenizer(runner.kv)
    template = None
    if tokenizer.chat_template and not args.no_chat_template:
        template = ChatTemplate(tokenizer.chat_template, tokenizer.bos_token, tokenizer.eos_token)
    kind = "DFlash2" if drafter else f"MTP ({mtp.name})" if mtp else "MTP"
    drafting = f", {runner.speculative} {kind} drafts per step" if runner.speculative else ""
    drafting += f", copied blocks of {runner.copy}" if runner.copy else ""
    tuned = f", BEAM {os.environ['JITBEAM']} for tinygrad's kernels" if os.environ.get("JITBEAM") else ""
    saved = runner.session.programs
    if saved.restored:
        doing = f"{len(saved.loaded)} programs loaded from {saved.path} ({saved.seconds:.1f}s), running each once"
    else:
        doing = f"compiling (programs {saved.note})" if saved.note else "compiling"
    log.info(
        "loaded %s in %.1fs%s; %s (prefill chunks of %d%s)",
        path.name,
        time.perf_counter() - start,
        drafting,
        doing,
        chunk,
        tuned,
    )
    runner.warmup()
    log.info("%s in %.1fs", "ready" if saved.restored else "compiled", runner.warmup_seconds)
    log.info("  %s", ", ".join(f"{name} {seconds:.0f}s" for name, seconds in runner.warmup_stages.items()))
    if not saved.restored and saved.path is not None:
        if saved.note:
            log.warning("programs %s: the next start compiles them again", saved.note)
        else:
            log.info(
                "programs saved to %s (%.0f MB, %.1fs): the next start loads them",
                saved.path,
                saved.bytes / 1e6,
                saved.seconds,
            )
    if runner.checkpoints.slots:
        size = runner.session.checkpoint_bytes / 2**20
        log.info("%d state checkpoints of %.0f MiB for prompt reuse", runner.checkpoints.slots, size)

    engine = Engine(runner, max_waiting=args.max_waiting, prefill_slice=chunk * args.prefill_chunks)
    name = args.name or str(runner.kv.get("general.name") or path.stem)
    config = ServerConfig(name, runner.max_context, sampling_defaults(runner.kv), args.api_key, args.queue_timeout)
    app = create_app(EngineClient(engine), tokenizer, template, config)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", lifespan="off"))
    http = threading.Thread(target=server.run, name="http", daemon=True)
    http.start()
    log.info("serving %s on http://%s:%d/v1 (context %d)", name, args.host, args.port, runner.max_context)
    signal.signal(signal.SIGTERM, lambda *_: engine.stop())  # what service managers send; Ctrl-C raises below
    try:
        engine.run()  # tinygrad's devices and JIT stay on this thread
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        server.should_exit = True
        http.join(timeout=10)
    return 0


def add_parser(sub: Any) -> None:
    sp = sub.add_parser("serve", help="serve a model through an OpenAI-compatible HTTP API")
    sp.add_argument("model", help="GGUF path, or a file name under the model directory")
    sp.add_argument("--device", default=None, help="auto, amd, cpu or a tinygrad DEV string (default: auto)")
    sp.add_argument("--ctx", type=int, default=32768, help="context tokens (default: %(default)s)")
    sp.add_argument("--host", default="127.0.0.1", help="address to listen on (default: %(default)s)")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--api-key", default=None, help="require this bearer token on /v1 routes")
    sp.add_argument("--name", default=None, help="model name reported by /v1/models (default: from the file)")
    sp.add_argument(
        "--chunk",
        type=int,
        default=None,
        help=f"prompt tokens per prefill call (default: {DEFAULT_CHUNK}; 2048 for Qwen3.8-Flash-Next)",
    )
    sp.add_argument(
        "--beam",
        type=int,
        default=None,
        metavar="N",
        help="tune tinygrad's generic kernels by BEAM search of width N, cached after the first start; 0 to skip "
        "(default: 2 for Qwen3.8-Flash-Next, else 0; JITBEAM in the environment takes precedence)",
    )
    sp.add_argument(
        "--prefill-chunks", type=int, default=4, help="prefill calls between cancellation checks (default: %(default)s)"
    )
    sp.add_argument(
        "--max-waiting", type=int, default=16, help="queued requests beyond which 429 (default: %(default)s)"
    )
    sp.add_argument("--queue-timeout", type=float, default=None, help="seconds a request may wait before it starts")
    sp.add_argument("--no-chat-template", action="store_true", help="serve /v1/completions only")
    sp.add_argument(
        "--speculative",
        type=int,
        default=None,
        metavar="K",
        help=f"drafts per decode step, 0 to turn speculative decoding off (default: {DFLASH_DRAFTS} with "
        f"--draft-model, else {DEFAULT_DRAFTS} when the model has an MTP block; 4 either way for Qwen3.8-Flash-Next)",
    )
    sp.add_argument(
        "--mtp",
        default=None,
        metavar="GGUF",
        help="the MTP block's own GGUF for a model that ships it apart (Qwen3.8-Flash-Next's mtp-*.gguf); by default "
        "one found next to the model, in the directory above, or in an MTP directory of either",
    )
    sp.add_argument(
        "--copy",
        type=int,
        default=None,
        metavar="N",
        help="also draft blocks of N tokens copied from earlier text of the conversation, when the last tokens "
        "occurred before and the MTP block drafts what followed (Qwen3.8-Flash-Next with its MTP block; N + 1 a "
        "multiple of 16; 0: off)",
    )
    sp.add_argument(
        "--draft-model",
        default=None,
        metavar="GGUF",
        help="a DFlash2 drafter made for the model (e.g. Qwen3.8-27B-DFlash2-Q4_K_M.gguf) instead of its MTP block",
    )
    sp.add_argument(
        "--program-cache",
        default=None,
        metavar="DIR",
        help="keep the compiled programs in DIR, so a start with the same model, settings and sources loads them "
        "instead of compiling for minutes; 0 to compile at every start (default: TINYLLM_PROGRAM_CACHE, or "
        "tinyllm/programs in the user's cache directory)",
    )
    sp.add_argument(
        "--state-checkpoints",
        type=int,
        default=CHECKPOINTS,
        metavar="N",
        help="copies of the recurrent state kept so a prompt that departs from the previous one (an agent's reply "
        "re-rendered by the chat template) resumes near that point; 0 to reuse only exact extensions "
        "(default: %(default)s)",
    )
    sp.set_defaults(fn=main)
