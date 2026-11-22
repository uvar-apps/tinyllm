"""The memory plan: what a serving configuration needs, and whether the machine has it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tinyllm.models.spec import StateLayout

GIB = 1 << 30
DEFAULT_SCRATCH = 2 * GIB  # logits, activations, drafter weights
HEADROOM = 0.95  # plans above this share of the limit are refused


@dataclass(frozen=True)
class MemoryPlan:
    weights: int
    slots: int
    context: int
    slot_bytes: int  # one slot: KV cache at `context` tokens plus recurrent state
    state_store: int
    scratch: int
    host_total: int | None
    gpu_mappable: int | None

    @property
    def total(self) -> int:
        return self.weights + self.slots * self.slot_bytes + self.state_store + self.scratch

    @property
    def limit(self) -> int | None:
        limits = [x for x in (self.host_total, self.gpu_mappable) if x]
        return min(limits) if limits else None

    @property
    def fits(self) -> bool | None:
        return None if self.limit is None else self.total <= self.limit * HEADROOM


def host_memory(meminfo: Path = Path("/proc/meminfo")) -> int | None:
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def gpu_mappable_memory(drm: Path = Path("/sys/class/drm")) -> int | None:
    """Largest GTT size among AMD cards: on unified-memory machines this caps what the GPU can address."""
    best: int | None = None
    for f in drm.glob("card*/device/mem_info_gtt_total"):
        try:
            best = max(best or 0, int(f.read_text()))
        except (OSError, ValueError):
            continue
    return best


def plan(
    layout: StateLayout,
    weights: int,
    context: int,
    slots: int,
    state_store: int | None = None,
    scratch: int = DEFAULT_SCRATCH,
    host_total: int | None = None,
    gpu_mappable: int | None = None,
) -> MemoryPlan:
    """Size a configuration. The state store defaults to room for one parked sequence per slot."""
    slot = layout.slot_bytes(context)
    return MemoryPlan(
        weights=weights,
        slots=slots,
        context=context,
        slot_bytes=slot,
        state_store=slots * slot if state_store is None else state_store,
        scratch=scratch,
        host_total=host_total,
        gpu_mappable=gpu_mappable,
    )
