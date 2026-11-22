"""Locating model files and identifying them cheaply."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENV_MODEL_DIR = "TINYLLM_MODEL_DIR"
_EDGE = 1 << 20


def model_dir() -> Path:
    return Path(os.environ.get(ENV_MODEL_DIR, "~/.cache/models")).expanduser()


def find_model(ref: str | Path) -> Path:
    """Resolve a path, or a bare file name searched for under the model directory."""
    path = Path(ref).expanduser()
    if path.exists():
        return path
    if path.parent == Path("."):
        matches = sorted(model_dir().rglob(path.name)) if model_dir().is_dir() else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"{path.name} is ambiguous under {model_dir()}: {[str(m) for m in matches]}")
    raise FileNotFoundError(f"model {ref} not found (searched {model_dir()} for bare names)")


def fingerprint(path: str | Path) -> str:
    """A fast identity for a multi-gigabyte file: SHA-256 over its size and its first and last megabyte."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(_EDGE))
        if size > 2 * _EDGE:
            f.seek(size - _EDGE)
            h.update(f.read(_EDGE))
    return h.hexdigest()[:16]
