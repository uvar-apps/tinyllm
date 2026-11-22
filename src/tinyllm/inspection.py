"""`tinyllm inspect`: what a GGUF file contains and what serving it would cost. Opens no device."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinyllm.models import memory, spec
from tinyllm.weights import gguf, quant

_BLOCK_RE = re.compile(r"^blk\.\d+\.")
# 2-D weights that are looked up or applied elementwise, not multiplied as a linear layer
_NOT_LINEAR = ("token_embd.weight", "per_layer_token_embd.weight")


def _is_linear(t: gguf.TensorInfo) -> bool:
    return len(t.shape) == 2 and t.name.endswith(".weight") and t.name not in _NOT_LINEAR and "conv1d" not in t.name


def report(path: str | Path, context: int, slots: int, state_store: int | None = None) -> dict[str, Any]:
    g = gguf.read(path)
    s = spec.from_gguf(g)
    params = sum(t.elements for t in g.tensors)

    by_type: Counter[str] = Counter()
    for t in g.tensors:
        by_type[t.qtype.name] += t.elements

    # which linear layers would miss the packed-weight kernels and fall back to a slow path
    linear = [t for t in g.tensors if _is_linear(t)]
    slow = [t for t in linear if quant.kernel_path(t.qtype, t.shape[0], t.shape[1]) == quant.GENERIC]
    slow_families: Counter[str] = Counter()
    for t in slow:
        slow_families[f"{_BLOCK_RE.sub('blk.N.', t.name)} [{t.qtype.name}]"] += t.elements
    linear_params = sum(t.elements for t in linear)

    out: dict[str, Any] = {
        "file": str(g.path),
        "parts": len(g.parts),
        "file_bytes": g.file_bytes,
        "tensor_bytes": g.tensor_bytes,
        "tensors": len(g.tensors),
        "parameters": params,
        "bits_per_weight": 8 * g.tensor_bytes / params if params else 0.0,
        "spec": asdict(s),
        "types": dict(by_type.most_common()),
        "undecodable_types": sorted({t.qtype.name for t in g.tensors if not t.qtype.upstream}),
        "linear_parameters": linear_params,
        "fast_kernel_share": 1 - sum(t.elements for t in slow) / linear_params if linear_params else 0.0,
        "slow_linear_families": dict(slow_families.most_common()),
        "sampling_defaults": {k.split(".")[-1]: v for k, v in g.kv.items() if k.startswith("general.sampling.")},
        "plan": None,
    }
    if s.state is not None:
        context = min(context, s.context_length) if s.context_length else context
        p = memory.plan(
            s.state,
            g.tensor_bytes,
            context,
            slots,
            state_store,
            host_total=memory.host_memory(),
            gpu_mappable=memory.gpu_mappable_memory(),
        )
        out["plan"] = {**asdict(p), "total": p.total, "limit": p.limit, "fits": p.fits}
    return out


def _gb(n: float) -> str:
    return f"{n / 1e9:8.2f} GB"


def render(r: dict[str, Any]) -> str:
    s = r["spec"]
    lines = [
        f"{s['name']}  ({s['architecture']})",
        f"  file          {r['file']}" + (f"  ({r['parts']} parts)" if r["parts"] > 1 else ""),
        f"  size          {_gb(r['file_bytes']).strip()}, {r['parameters'] / 1e9:.2f} B parameters"
        f" in {r['tensors']} tensors, {r['bits_per_weight']:.2f} bits per weight",
        f"  shape         {s['blocks']} blocks"
        + (f" + {s['mtp_blocks']} MTP" if s["mtp_blocks"] else "")
        + f", width {s['dim']}, feed-forward {s['feed_forward']}, vocabulary {s['vocab_size']}",
        f"  attention     {s['heads']} heads, {s['kv_heads']} KV heads, head dimension {s['head_dim']}"
        + (f", {s['experts']} experts" if s["experts"] else ""),
        f"  context       {s['context_length']}",
    ]
    if r["sampling_defaults"]:
        lines.append("  sampling      " + ", ".join(f"{k} {v:g}" for k, v in sorted(r["sampling_defaults"].items())))

    lines.append("\nStorage types (share of parameters)")
    for name, n in r["types"].items():
        lines.append(f"  {name:8s} {100 * n / r['parameters']:5.1f}%   {quant.BY_NAME[name].bits_per_weight:5.2f} bpw")
    if r["undecodable_types"]:
        lines.append(f"  NOT decodable by the pinned tinygrad: {', '.join(r['undecodable_types'])}")

    lines.append(f"\nLinear layers on fast AMD kernels: {100 * r['fast_kernel_share']:.1f}% of their parameters")
    for family, n in r["slow_linear_families"].items():
        lines.append(f"  generic path: {family}  ({n / 1e6:.0f} M parameters)")

    st = s["state"]
    if st is None:
        lines.append(f"\nState layout for '{s['architecture']}' is not modelled yet; no memory plan.")
        return "\n".join(lines)
    lines += [
        "\nPer-sequence state",
        f"  layers        {st['attention_layers']} attention, {st['recurrent_layers']} recurrent",
        f"  KV cache      {st['kv_bytes_per_token'] / 1024:.0f} KiB per token",
        f"  checkpoint    {(st['recurrent_bytes'] + st['conv_bytes']) / (1 << 20):.0f} MiB"
        " (recurrent state and convolution windows)",
    ]
    p = r["plan"]
    lines += [
        f"\nMemory plan: {p['slots']} slots at {p['context']} tokens",
        f"  weights      {_gb(p['weights'])}",
        f"  slot state   {_gb(p['slots'] * p['slot_bytes'])}   ({p['slots']} x {_gb(p['slot_bytes']).strip()})",
        f"  state store  {_gb(p['state_store'])}",
        f"  scratch      {_gb(p['scratch'])}",
        f"  total        {_gb(p['total'])}",
    ]
    if p["limit"] is not None:
        verdict = "fits" if p["fits"] else "DOES NOT FIT"
        lines.append(
            f"  available    {_gb(p['limit'])}   -> {verdict} ({100 * p['total'] / p['limit']:.0f}% of the limit)"
        )
    return "\n".join(lines)
