"""`tinyllm serve` end to end on the GPU, driven through the OpenAI Python SDK (tools/openai_sdk_check.py)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from conftest import REPO


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _serve_and_check(model: Path, programs: Path) -> str:
    """Start the server on `model`, run the SDK checks against it, stop it; its log."""
    env = {k: v for k, v in os.environ.items() if k not in ("TINYLLM_DEVICE", "DEV")}
    port = _free_port()
    cmd = [sys.executable, "-m", "tinyllm.cli", "serve", str(model), "--ctx", "8192", "--port", str(port)]
    serve = subprocess.Popen(
        [*cmd, "--program-cache", str(programs)],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            assert serve.poll() is None, serve.stdout.read() if serve.stdout else "server exited"
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        url = f"http://127.0.0.1:{port}/v1"
        check = subprocess.run(
            [sys.executable, "tools/openai_sdk_check.py", "--base-url", url, "--skip", "speed"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert check.returncode == 0, check.stdout + check.stderr
        assert check.stdout.count("PASS") == 6
    finally:
        serve.terminate()  # SIGTERM: the server stops its engine and exits cleanly
        out, _ = serve.communicate(timeout=60)
        assert serve.returncode == 0, out[-3000:]
    return out


@pytest.mark.gpu
def test_serve_passes_the_sdk_checks_with_programs_built_and_loaded(dev_model: Path, tmp_path: Path) -> None:
    """The first start compiles its programs and saves them; the second loads them (tinyllm.tg.programs) and passes
    the same checks."""
    first = _serve_and_check(dev_model, tmp_path)
    assert "programs saved to" in first, first[-3000:]
    second = _serve_and_check(dev_model, tmp_path)
    assert "programs loaded from" in second and "compiled in" not in second, second[-3000:]
