"""llama.cpp on the CPU as an independent reference implementation.

It reads the same GGUF file through an unrelated code base, so agreement with it rules out
errors that tinygrad's engine and anything derived from it would share. The server is started
on a free local port, used, and stopped; it never touches the GPU.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinyllm.oracle.record import OracleRecord

ENV_SERVER = "TINYLLM_REF_LLAMA_SERVER"


def default_server() -> Path:
    if env := os.environ.get(ENV_SERVER):
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[3] / ".ref" / "llama.cpp" / "build" / "bin" / "llama-server"


class ReferenceServer:
    """Context manager around a `llama-server` process."""

    def __init__(self, model: Path, binary: Path | None = None, context: int = 2048, threads: int = 16) -> None:
        self.binary = binary or default_server()
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not found; run tools/build_ref_llamacpp.sh or set {ENV_SERVER}")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = int(s.getsockname()[1])
        self.cmd = [str(self.binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port)]
        self.cmd += ["-c", str(context), "-t", str(threads), "-ngl", "0", "-np", "1"]
        self.proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> ReferenceServer:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}: {' '.join(self.cmd)}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.25)
        self.__exit__(None, None, None)
        raise RuntimeError("llama-server did not become ready within 180 s")

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            parsed: dict[str, Any] = json.loads(r.read())
            return parsed

    def tokenize(self, text: str) -> list[int]:
        return [
            int(t)
            for t in self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
        ]

    def next_token_logprobs(self, tokens: list[int], top: int = 8) -> dict[int, float]:
        """Log probabilities of the `top` most likely next tokens after `tokens`."""
        body = {"prompt": tokens, "n_predict": 1, "temperature": 0.0, "n_probs": top, "cache_prompt": True}
        probs = self._post("/completion", body).get("completion_probabilities") or []
        if not probs:
            raise RuntimeError("llama-server returned no token probabilities")
        entries = probs[0].get("top_logprobs") or probs[0].get("top_probs") or []
        return {int(e["id"]): float(e["logprob"]) for e in entries if "logprob" in e}


@dataclass
class ReferenceReport:
    steps: int = 0
    top1_agree: int = 0
    in_reference_top: int = 0
    logprob_diffs: list[float] = field(default_factory=list)
    tokenizer_mismatches: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)

    @property
    def top1_rate(self) -> float:
        return self.top1_agree / self.steps if self.steps else 0.0

    def summary(self) -> str:
        diffs = sorted(self.logprob_diffs)
        mean = sum(diffs) / len(diffs) if diffs else float("nan")
        worst = diffs[-1] if diffs else float("nan")
        lines = [
            f"{self.steps} positions compared against llama.cpp",
            f"  same most-likely token       {self.top1_agree}/{self.steps} ({self.top1_rate:.1%})",
            f"  our token in reference top-k {self.in_reference_top}/{self.steps}",
            f"  |log-probability difference| mean {mean:.4f}, max {worst:.4f}  (our token, where both report it)",
            f"  tokenizer                    {'identical' if not self.tokenizer_mismatches else 'DIFFERS'}",
        ]
        lines += [f"    tokenizer: {m}" for m in self.tokenizer_mismatches]
        lines += [f"    disagreement: {m}" for m in self.disagreements[:10]]
        return "\n".join(lines)


def compare(golden: OracleRecord, server: ReferenceServer, top: int = 8) -> ReferenceReport:
    """Walk each prompt's stored greedy path and ask llama.cpp for its next-token distribution at every position."""
    report = ReferenceReport()
    for p in golden.prompts:
        ref_ids = server.tokenize(p.text)
        if ref_ids != p.prompt_ids:
            report.tokenizer_mismatches.append(f"{p.id}: ours {len(p.prompt_ids)} tokens, reference {len(ref_ids)}")
        for i, step in enumerate(p.steps):
            ref = server.next_token_logprobs(p.prompt_ids + p.gen_ids[:i], top)
            ours = p.gen_ids[i]
            report.steps += 1
            ref_best = max(ref, key=lambda t: ref[t])
            if ref_best == ours:
                report.top1_agree += 1
            else:
                report.disagreements.append(f"{p.id}@{i}: ours {ours}, reference {ref_best}")
            if ours in ref:
                report.in_reference_top += 1
                ours_lp = step.logprob(ours)
                if ours_lp is not None:
                    report.logprob_diffs.append(abs(ours_lp - ref[ours]))
    return report
