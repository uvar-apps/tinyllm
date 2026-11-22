"""Device selection, decided before tinygrad is imported.

tinygrad reads the `DEV` environment variable once, at import. Its AMD device tries the
kernel-driver interface (KFD) first and can fall back to its own userspace PCI driver,
which unbinds the GPU from `amdgpu`. On a machine whose GPU drives the display that takes
the desktop down. This module therefore always names the interface explicitly and rejects
AMD targets that do not use KFD.

Nothing here imports tinygrad.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_DEVICE = "TINYLLM_DEVICE"
ENV_ALLOW_USERSPACE = "TINYLLM_ALLOW_AMD_USERSPACE"
KFD_NODE = Path("/dev/kfd")
KFD_TOPOLOGY = Path("/sys/devices/virtual/kfd/kfd/topology/nodes")

# gfx major versions tinygrad's AMD backend accepts: CDNA3/4 (9), RDNA3/3.5 (11), RDNA4 (12)
SUPPORTED_GFX_MAJORS = (9, 11, 12)
_ALIASES = {"amd": "KFD+AMD", "cpu": "CPU", "nv": "NV", "cuda": "CUDA"}


class DeviceConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AmdGpu:
    node: int
    name: str
    gfx_target: tuple[int, int, int]
    compute_units: int

    @property
    def arch(self) -> str:
        major, minor, step = self.gfx_target
        return f"gfx{major}{minor:x}{step:x}"

    @property
    def supported(self) -> bool:
        return self.gfx_target[0] in SUPPORTED_GFX_MAJORS


def detect_amd_gpus(topology: Path = KFD_TOPOLOGY) -> list[AmdGpu]:
    """AMD GPUs the kernel driver exposes for compute, read from the KFD topology in sysfs."""
    gpus: list[AmdGpu] = []
    if not topology.is_dir():
        return gpus
    for node in sorted((p for p in topology.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = dict(line.split()[:2] for line in (node / "properties").read_text().splitlines() if line.strip())
            version = int(props.get("gfx_target_version", "0"))
            if version == 0:  # CPU-only node
                continue
            name = (node / "name").read_text().strip() if (node / "name").exists() else ""
            simd, per_cu = int(props.get("simd_count", "0")), int(props.get("simd_per_cu", "1")) or 1
        except (OSError, ValueError):
            continue
        target = (version // 10000, (version // 100) % 100, version % 100)
        gpus.append(AmdGpu(int(node.name), name, target, simd // per_cu))
    return gpus


def _targets(dev: str) -> list[tuple[str, str]]:
    """(interface, device) for each `;`-separated target of a tinygrad DEV string.

    A target is `[IFACE[:indices]+]DEVICE[:RENDERER[:ARCH]]`.
    """
    out: list[tuple[str, str]] = []
    for part in dev.split(";"):
        iface, _, rest = part.rpartition("+")
        out.append((iface.split(":")[0].upper(), rest.split(":")[0].upper()))
    return out


def harden(dev: str, environ: Mapping[str, str] = os.environ) -> str:
    """Return `dev` with every AMD target pinned to the KFD interface.

    An AMD target with no interface gets KFD. An AMD target naming another interface
    (PCI, USB) is an error unless TINYLLM_ALLOW_AMD_USERSPACE=1.
    """
    parts: list[str] = []
    for part, (iface, device) in zip(dev.split(";"), _targets(dev), strict=True):
        if device == "" and iface == "":
            raise DeviceConfigError(f"empty target in DEV string {dev!r}")
        # a target with no device name applies to every device, AMD included
        if device not in ("AMD", "") or iface == "KFD":
            parts.append(part)
        elif iface == "":
            parts.append("KFD+" + part.lstrip("+"))
        elif environ.get(ENV_ALLOW_USERSPACE) == "1":
            parts.append(part)
        else:
            raise DeviceConfigError(
                f"DEV target {part!r} uses tinygrad's userspace {iface} driver, which unbinds the GPU from the "
                f"kernel driver and takes a display GPU down. Use KFD+AMD, or set {ENV_ALLOW_USERSPACE}=1 "
                f"on a machine whose GPU is dedicated to compute."
            )
    return ";".join(parts)


def resolve(
    preference: str = "auto",
    environ: Mapping[str, str] = os.environ,
    topology: Path = KFD_TOPOLOGY,
    kfd_node: Path = KFD_NODE,
) -> str:
    """Turn a device preference (`auto`, `amd`, `cpu`, `nv`, `cuda` or a raw tinygrad DEV string) into a DEV value."""
    pref = preference.strip()
    key = pref.lower()
    if key == "auto":
        usable = kfd_node.exists() and os.access(kfd_node, os.R_OK | os.W_OK)
        return "KFD+AMD" if usable and any(g.supported for g in detect_amd_gpus(topology)) else "CPU"
    if key == "amd":
        if not kfd_node.exists():
            raise DeviceConfigError(f"--device amd needs the amdgpu kernel driver ({kfd_node} is missing)")
        return "KFD+AMD"
    if key in _ALIASES:
        return _ALIASES[key]
    if not pref:
        raise DeviceConfigError("empty device preference")
    return harden(pref, environ)


_configured: str | None = None


def configure(preference: str | None = None) -> str:
    """Decide the tinygrad device and export it as `DEV`. Must run before tinygrad is imported.

    Precedence: the argument, then TINYLLM_DEVICE, then an existing DEV, then `auto`.
    Calling it again returns the first decision; asking for a different device raises.
    """
    global _configured
    explicit = preference if preference is not None else os.environ.get(ENV_DEVICE)
    wanted = resolve(explicit) if explicit else harden(os.environ["DEV"]) if os.environ.get("DEV") else resolve("auto")
    if _configured is not None:
        if preference is not None and wanted != _configured:
            raise DeviceConfigError(f"device already configured as {_configured!r}; cannot switch to {wanted!r}")
        return _configured
    if "tinygrad" in sys.modules and os.environ.get("DEV", "") != wanted:
        raise DeviceConfigError(
            f"tinygrad was imported before tinyllm chose a device (DEV={os.environ.get('DEV', '')!r}, wanted "
            f"{wanted!r}). Import tinyllm.tg first, or set DEV yourself."
        )
    os.environ["DEV"] = wanted
    _configured = wanted
    return wanted
