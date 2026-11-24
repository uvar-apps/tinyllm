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


def has_mtp(path: Path) -> bool:
    """Whether the GGUF carries one MTP (nextn) block, which speculative decoding drafts with."""
    from tinyllm.weights import gguf

    kv = gguf.read(path, header_only=True).kv
    return int(kv.get(f"{kv.get('general.architecture')}.nextn_predict_layers", 0) or 0) == 1


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    from tinyllm import device

    device.configure(args.device)
    import uvicorn

    from tinyllm.engine.core import Engine
    from tinyllm.frontend.template import ChatTemplate
    from tinyllm.runner.upstream import UpstreamRunner
    from tinyllm.server.app import ServerConfig, create_app
    from tinyllm.server.client import EngineClient
    from tinyllm.tg.tokenizer import GGUFTokenizer
    from tinyllm.weights.resolve import find_model

    path = find_model(args.model)
    start = time.perf_counter()
    drafter = Path(args.draft_model).expanduser() if args.draft_model else None
    default = (DFLASH_DRAFTS if drafter else DEFAULT_DRAFTS) if drafter or has_mtp(path) else 0
    k = args.speculative if args.speculative is not None else default
    runner = UpstreamRunner(
        path,
        args.ctx,
        chunk=args.chunk,
        speculative=k,
        drafter=drafter if k else None,
        checkpoints=args.state_checkpoints,
    )
    tokenizer = GGUFTokenizer(runner.kv)
    template = None
    if tokenizer.chat_template and not args.no_chat_template:
        template = ChatTemplate(tokenizer.chat_template, tokenizer.bos_token, tokenizer.eos_token)
    kind = "DFlash2" if drafter else "MTP"
    drafting = f", {runner.speculative} {kind} drafts per step" if runner.speculative else ""
    log.info("loaded %s in %.1fs%s; compiling", path.name, time.perf_counter() - start, drafting)
    runner.warmup()
    log.info("compiled in %.1fs", runner.warmup_seconds)
    if runner.checkpoints.slots:
        size = runner.session.checkpoint_bytes / 2**20
        log.info("%d state checkpoints of %.0f MiB for prompt reuse", runner.checkpoints.slots, size)

    engine = Engine(runner, max_waiting=args.max_waiting, prefill_slice=args.chunk * args.prefill_chunks)
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
    sp.add_argument("--chunk", type=int, default=128, help="prompt tokens per prefill call (default: %(default)s)")
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
        f"--draft-model, else {DEFAULT_DRAFTS} when the model has an MTP block)",
    )
    sp.add_argument(
        "--draft-model",
        default=None,
        metavar="GGUF",
        help="a DFlash2 drafter (e.g. Qwen3.8-27B-DFlash2-Q4_K_M.gguf) instead of the model's own MTP block",
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
