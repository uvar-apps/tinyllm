"""The only package that imports tinygrad.

Importing it fixes the device (see `tinyllm.device`), verifies the tinygrad pin
(see `tinyllm.pin`) and then imports tinygrad. Everything else in tinyllm reaches
tinygrad through the names re-exported here, so a tinygrad bump touches one package.
"""

from __future__ import annotations

import time
import types
from typing import Any

from tinyllm import device as _device
from tinyllm import pin as _pin

DEV: str = _device.configure()
TINYGRAD_REV: str | None = _pin.check()

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, UOp, dtypes, nn  # noqa: E402
from tinygrad.device import Buffer, BufferSpec  # noqa: E402

__all__ = [
    "DEV",
    "TINYGRAD_REV",
    "Buffer",
    "BufferSpec",
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
    "set_wait",
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


YIELD_SPIN_SECONDS = 0.0005  # a "yield" wait polls without pause this long after the last progress, then sleeps
YIELD_SLEEP_SECONDS = 0.00005


def _yielding_wait(self: Any, sig: Any, value: int, timeout: int | None = None) -> None:
    """tinygrad's HCQ signal wait (`HCQCompiled._wait_signal` at the pinned commit) with short sleeps.

    The stock loop polls in Python and holds the interpreter lock for the whole wait, so another Python thread (the
    HTTP server) can only run when the interpreter forces a switch, every 5 ms. This one polls for
    YIELD_SPIN_SECONDS and then sleeps in YIELD_SLEEP_SECONDS steps, releasing the lock; completion is seen within
    about 0.1 ms. Timeout and fault detection are unchanged.
    """
    timeout = timeout if timeout is not None and self.can_recover else None
    limit = (timeout or self.wait_timeout_ms) / 1000
    start, done, err = time.perf_counter(), sig[0], self.error_state.host.view(fmt="q")
    while done < value and not err[0]:
        if done != (done := sig[0]):
            start = time.perf_counter()
            continue
        elapsed = time.perf_counter() - start
        if elapsed > limit:
            raise RuntimeError(f"{self.device} signal wait timed out")
        if elapsed > YIELD_SPIN_SECONDS:
            time.sleep(YIELD_SLEEP_SECONDS)
    if err[0]:
        raise RuntimeError(f"{self.device} failed with {err[0]}")


def set_wait(mode: str) -> None:
    """How the host waits for the GPU (plan, phase 1 item 8):

    "poll"   poll only (`set_polling_wait`): lowest latency; holds the interpreter lock and one CPU core.
    "yield"  poll briefly, then sleep in 50 microsecond steps: for the server, where an HTTP thread shares the
             interpreter; costs at most about 0.1 ms per wait.
    "sleep"  tinygrad's own: poll for 200 ms, then sleep on a kernel event that on Strix Halo only wakes the
             waiter every 208 ms (docs/01-research-findings.md, section 5.3). Benchmarks must not use it.
    """
    dev = Device[Device.DEFAULT]
    if not hasattr(dev, "_wait_signal"):
        return  # not an HCQ device (the CPU in tests): nothing to change
    if mode not in ("poll", "yield", "sleep"):
        raise ValueError(f"unknown wait mode {mode!r}")
    set_polling_wait(mode != "sleep")
    if mode == "yield":
        dev._wait_signal = types.MethodType(_yielding_wait, dev)
    else:
        dev.__dict__.pop("_wait_signal", None)  # back to the class's own loop


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
