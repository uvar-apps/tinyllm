"""The only package that imports tinygrad.

Importing it fixes the device (see `tinyllm.device`), verifies the tinygrad pin
(see `tinyllm.pin`) and then imports tinygrad. Everything else in tinyllm reaches
tinygrad through the names re-exported here, so a tinygrad bump touches one package.
"""

from __future__ import annotations

from typing import Any

from tinyllm import device as _device
from tinyllm import pin as _pin

DEV: str = _device.configure()
TINYGRAD_REV: str | None = _pin.check()

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, UOp, dtypes, nn  # noqa: E402

__all__ = [
    "DEV",
    "TINYGRAD_REV",
    "Device",
    "GlobalCounters",
    "Tensor",
    "TinyJit",
    "UOp",
    "default_device",
    "device_info",
    "dtypes",
    "nn",
    "set_polling_wait",
    "synchronize",
]


def default_device() -> str:
    return str(Device.DEFAULT)


def synchronize() -> None:
    """Block until everything queued on the default device has finished."""
    Device[Device.DEFAULT].synchronize()


def set_polling_wait(enabled: bool = True) -> None:
    """Wait for GPU completion by polling only, instead of tinygrad's poll-then-sleep.

    tinygrad's AMD device polls for 200 ms and then sleeps in 200 ms steps on a kernel event. On
    Strix Halo that event does not wake the waiter, so a step longer than 200 ms is observed as
    the next multiple of about 208 ms (docs/01-research-findings.md, section 5.3). Polling keeps
    one CPU thread busy while the GPU works, which tinygrad already does for every step shorter
    than 200 ms. The cost: a GPU fault is noticed at the wait timeout instead of within 200 ms.
    No effect on devices that never sleep.
    """
    dev = Device[Device.DEFAULT]
    default = getattr(type(dev), "sleep_timeout_ms", None)
    if default is not None:
        dev.sleep_timeout_ms = None if enabled else default


def device_info() -> dict[str, Any]:
    """Facts about the default device for logs and benchmark records. Opens the device."""
    dev = Device[Device.DEFAULT]
    info: dict[str, Any] = {"device": default_device(), "dev_env": DEV, "renderer": type(dev.renderer).__name__}
    for attr in ("arch", "target", "cu_cnt", "wave_cnt"):
        if (value := getattr(dev, attr, None)) is not None:
            info[attr] = list(value) if isinstance(value, tuple) else value
    if (iface := getattr(dev, "iface", None)) is not None:
        info["interface"] = type(iface).__name__
    return info
