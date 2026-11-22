"""Shared test setup.

Tests run on the CPU so they behave the same on a developer machine and in CI. tinygrad fixes
its device at import, so the choice is made here, before any test imports `tinyllm.tg`.
Tests that need the GPU are marked `gpu` and drive the CLI in a subprocess instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["TINYLLM_DEVICE"] = "cpu"
os.environ.pop("DEV", None)

from tinyllm.weights.resolve import model_dir

DEV_MODEL = "Qwen3.5-0.8B-Q8_0.gguf"
GOLDEN_DIR = Path(__file__).parent / "golden"
REPO = Path(__file__).resolve().parents[1]


def _find(name: str) -> Path | None:
    matches = sorted(model_dir().rglob(name)) if model_dir().is_dir() else []
    return matches[0] if matches else None


@pytest.fixture(scope="session")
def dev_model() -> Path:
    path = _find(DEV_MODEL)
    if path is None:
        pytest.skip(f"{DEV_MODEL} not found under {model_dir()}; run tools/fetch_dev_models.sh")
    return path


@pytest.fixture(scope="session")
def reference_server() -> Path:
    from tinyllm.oracle.llamacpp import default_server

    binary = default_server()
    if not binary.exists():
        pytest.skip("reference llama.cpp not built; run tools/build_ref_llamacpp.sh")
    return binary
