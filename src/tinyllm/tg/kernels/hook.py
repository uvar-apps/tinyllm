"""Routes linear layers of tinygrad.llm models to tinyllm's kernels, and one-token attention to tinyllm's.

`install()` wraps `tinygrad.llm.kernels.amd.Linear.__call__` in this process. At a layer's first call
the wrapper checks whether the layer's weights are something a tinyllm kernel serves; if not, or if
that kernel is not enabled, the layer keeps tinygrad's own path. The decision is stored on the layer.
With "attention" enabled it also wraps the model's `flash_attention`, so one-token calls (decoding) run on
`tinyllm.tg.kernels.decode_mma`, which the speculative verify then uses too.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

from tinygrad import Tensor
from tinygrad.llm import model as llm_model
from tinygrad.llm.kernels import amd

from tinyllm.tg.kernels import decode_mma, dense, densemm, iq4xs, q4, q6k, q8, q45k

ENV = "TINYLLM_KERNELS"  # "all" (default), "none", or a comma-separated subset of KERNELS
KERNELS = ("dense", "dense_mm", "q4", "q45k", "iq4xs", "q6k", "q8", "attention")
Run = Callable[[Any, Tensor, Callable[[Any, Tensor], Tensor]], Tensor]  # (layer, x, tinygrad's own call)
# prefill-only kernels: they take over tinygrad's packed weights and hand the rows they do not serve back to it
_PREFILL: dict[str, tuple[Callable[[Any], Any], Run]] = {
    "q45k": (q45k.find_q45k, q45k.q45k_linear),
    "iq4xs": (iq4xs.find_iq4xs, iq4xs.iq4xs_linear),
    "q6k": (q6k.find_q6k, q6k.q6k_linear),
    "q8": (q8.find_q8, q8.q8_linear),
}
_enabled: set[str] = set()


def _dense(layer: Any, x: Tensor, fallback: Callable[[Any, Tensor], Tensor]) -> Tensor:
    """BF16/F16 layers: matrix cores for the rows dense_mm serves, the decode kernel for the others."""
    if "dense_mm" in _enabled and densemm.serves(layer, x):
        return densemm.dense_mm_linear(layer, x)
    return dense.dense_linear(layer, x) if "dense" in _enabled else fallback(layer, x)


_RUN: dict[str, Run] = {
    "dense": _dense,
    "q4": lambda layer, x, fallback: q4.q4_linear(layer, x),
    **{name: run for name, (_, run) in _PREFILL.items()},
}
_UNSET = object()


def requested(environ: dict[str, str] | None = None) -> tuple[str, ...]:
    """The kernels TINYLLM_KERNELS asks for."""
    value = (os.environ if environ is None else environ).get(ENV, "all").strip().lower()
    if value in ("", "all", "1"):
        return KERNELS
    if value in ("none", "0"):
        return ()
    names = tuple(n.strip() for n in value.split(",") if n.strip())
    if unknown := [n for n in names if n not in KERNELS]:
        raise ValueError(f"{ENV}: unknown kernel(s) {unknown}; choose from {list(KERNELS)}, 'all' or 'none'")
    return names


def _eligible(layer: Any) -> bool:
    return bool(
        layer.ggml_type is None
        and layer.groups == 1
        and layer.weight_scale_inv is None
        and layer.in_features % 32 == 0
        and layer.weight.uop.axis is None
        and amd.amd_custom_kernels_supported(layer.weight.device)
    )


def _choose(layer: Any) -> str | None:
    layer._tinyllm_dense = layer._tinyllm_q4 = None
    if not _eligible(layer):
        return None
    if _enabled & {"dense", "dense_mm"} and (found := dense.find_dense(layer.weight)) is not None:
        layer._tinyllm_dense = found
        return "dense"
    if "q4" in _enabled and (found := q4.find_q4(layer.weight, layer.in_features)) is not None:
        layer._tinyllm_q4 = found
        return "q4"
    for name, (find, _) in _PREFILL.items():
        if name in _enabled and (found := find(layer)) is not None:
            # take over tinygrad's own packing so its decode kernel still serves the rows this kernel does not
            layer.ggml_type, layer.weight, layer.shard_axis = found[0], found[1], None
            return name
    return None


def install(names: Iterable[str] = KERNELS) -> None:
    """Enable the named kernels (all by default) and install the hook once. Layers already called keep their choice."""
    names = tuple(names)
    if unknown := [n for n in names if n not in KERNELS]:
        raise ValueError(f"unknown kernel(s) {unknown}; choose from {list(KERNELS)}")
    _enabled.update(names)
    if "attention" in names and not hasattr(llm_model.flash_attention, "_tinyllm_original"):
        llm_model.flash_attention = decode_mma.decode_attention(llm_model.flash_attention)
    original = amd.Linear.__call__
    if getattr(original, "_tinyllm_hook", False):
        return

    def __call__(self: Any, x: Tensor) -> Tensor:
        kernel = self.__dict__.get("_tinyllm_kernel", _UNSET)
        if kernel is _UNSET:
            kernel = self._tinyllm_kernel = _choose(self)
        if kernel is None:
            return original(self, x)
        run = _RUN[kernel]  # rows a kernel does not serve go to tinygrad's own path (`original`)
        if isinstance(x.numel(), int):
            return run(self, x, original)
        # symbolic token count: run on the padded maximum and slice the garbage rows off
        out = run(self, x.pad_to(x.max_shape), original)
        return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))

    __call__._tinyllm_hook = True  # type: ignore[attr-defined]
    __call__.__wrapped__ = original  # type: ignore[attr-defined]
    amd.Linear.__call__ = __call__


def installed() -> tuple[str, ...]:
    """The kernels enabled in this process, in a fixed order; empty when the hook is not installed."""
    if not getattr(amd.Linear.__call__, "_tinyllm_hook", False):
        return ()
    return tuple(k for k in KERNELS if k in _enabled)
