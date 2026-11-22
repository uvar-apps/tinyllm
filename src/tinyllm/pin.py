"""The tinygrad commit tinyllm is developed and benchmarked against.

tinygrad/llm changes daily and renames internals without notice, so tinyllm refuses to
run on any other commit unless explicitly told to. Bump TINYGRAD_REV together with the
`rev` in pyproject.toml, then rerun the oracle and benchmark suites.
"""

from __future__ import annotations

import importlib.metadata
import json
import os

TINYGRAD_REV = "cdabdcba0747eae78b136cb79db1e12492441eda"
ENV_UNPINNED = "TINYLLM_UNPINNED_TINYGRAD"


class PinError(RuntimeError):
    pass


def installed_tinygrad_rev() -> str | None:
    """The git commit the installed tinygrad was built from, or None if it cannot be determined
    (for example an editable install from a local checkout)."""
    try:
        raw = importlib.metadata.distribution("tinygrad").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
    except json.JSONDecodeError:
        return None
    return commit if isinstance(commit, str) else None


def check() -> str | None:
    """Raise PinError unless the installed tinygrad is the pinned commit. Returns the installed commit."""
    rev = installed_tinygrad_rev()
    if rev == TINYGRAD_REV or os.environ.get(ENV_UNPINNED) == "1":
        return rev
    found = f"commit {rev[:12]}" if rev else "an install with no recorded commit"
    raise PinError(
        f"tinyllm is pinned to tinygrad {TINYGRAD_REV[:12]} but found {found}. "
        f"Run `uv sync`, or set {ENV_UNPINNED}=1 to run unpinned at your own risk."
    )
