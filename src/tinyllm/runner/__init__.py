"""Step programs behind the `Runner` interface (design 4.5). Phase 1 ships `UpstreamRunner`: one sequence through
tinygrad.llm's own model."""

from tinyllm.runner.base import Runner

__all__ = ["Runner"]
