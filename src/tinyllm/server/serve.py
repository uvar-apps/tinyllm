"""`tinyllm serve`: load a model, start the HTTP server on a thread, run the engine on the main thread."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

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
    k = args.speculative if args.speculative is not None else DEFAULT_DRAFTS if has_mtp(path) else 0
    runner = UpstreamRunner(path, args.ctx, chunk=args.chunk, speculative=k)
    tokenizer = GGUFTokenizer(runner.kv)
    template = None
    if tokenizer.chat_template and not args.no_chat_template:
        template = ChatTemplate(tokenizer.chat_template, tokenizer.bos_token, tokenizer.eos_token)
    drafting = f", {runner.speculative} MTP drafts per step" if runner.speculative else ""
    log.info("loaded %s in %.1fs%s; compiling", path.name, time.perf_counter() - start, drafting)
    runner.warmup()
    log.info("compiled in %.1fs", runner.warmup_seconds)

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
        help=f"MTP drafts per decode step, 0 to turn speculative decoding off "
        f"(default: {DEFAULT_DRAFTS} when the model has an MTP block)",
    )
    sp.set_defaults(fn=main)
