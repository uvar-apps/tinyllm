"""Writes small synthetic GGUF files for tests."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from tinyllm.weights import quant

_STRING, _ARRAY = 8, 9


def _string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def _type_of(v: Any) -> int:
    if isinstance(v, bool):
        return 7
    if isinstance(v, int):
        return 4 if 0 <= v < 2**32 else 11
    if isinstance(v, float):
        return 6
    if isinstance(v, str):
        return _STRING
    if isinstance(v, list):
        return _ARRAY
    raise TypeError(type(v))


def _value(v: Any, type_id: int) -> bytes:
    if type_id == 7:
        return struct.pack("<?", v)
    if type_id == 4:
        return struct.pack("<I", v)
    if type_id == 11:
        return struct.pack("<q", v)
    if type_id == 6:
        return struct.pack("<f", v)
    if type_id == _STRING:
        return _string(v)
    item_type = _type_of(v[0]) if v else 4
    return struct.pack("<IQ", item_type, len(v)) + b"".join(_value(x, item_type) for x in v)


def write_gguf(
    path: Path,
    kv: dict[str, Any],
    tensors: list[tuple[str, tuple[int, ...], str]],
    alignment: int = 32,
    truncate_data: int = 0,
) -> Path:
    """`tensors` holds (name, dims in GGUF order, storage type name). Tensor data is zeros."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, v in kv.items():
        t = _type_of(v)
        head += _string(key) + struct.pack("<I", t) + _value(v, t)
    offset = 0
    for name, dims, type_name in tensors:
        q = quant.BY_NAME[type_name]
        elements = 1
        for d in dims:
            elements *= d
        head += _string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", q.id, offset)
        offset += (q.nbytes(elements) + alignment - 1) // alignment * alignment
    pad = (-len(head)) % alignment
    path.write_bytes(head + b"\0" * pad + b"\0" * max(0, offset - truncate_data))
    return path


def qwen35_kv(blocks: int = 8, dim: int = 256, vocab: int = 100) -> dict[str, Any]:
    """Metadata for a miniature `qwen35` model: every 4th block is attention, the rest recurrent."""
    a = "qwen35"
    return {
        "general.architecture": a,
        "general.name": "tiny-qwen35",
        f"{a}.block_count": blocks + 1,
        f"{a}.nextn_predict_layers": 1,
        f"{a}.context_length": 4096,
        f"{a}.embedding_length": dim,
        f"{a}.feed_forward_length": 4 * dim,
        f"{a}.attention.head_count": 4,
        f"{a}.attention.head_count_kv": 2,
        f"{a}.attention.key_length": 64,
        f"{a}.full_attention_interval": 4,
        f"{a}.ssm.conv_kernel": 4,
        f"{a}.ssm.state_size": 32,
        f"{a}.ssm.group_count": 2,
        f"{a}.ssm.time_step_rank": 4,
        f"{a}.ssm.inner_size": 256,
        "general.sampling.temp": 1.0,
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(vocab)],
    }
