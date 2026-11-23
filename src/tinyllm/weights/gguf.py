"""GGUF header reader.

Reads metadata and the tensor directory only; tensor data is never touched and no device
is opened. Model files are untrusted input, so every length is checked against the file size
before it is used. Decoding tensor data stays with tinygrad (reached through `tinyllm.tg`).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from tinyllm.weights import quant

MAGIC = b"GGUF"
MAX_COUNT = 1 << 24  # sanity cap for kv, tensor and dimension counts
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class ArraySummary:
    """Stands in for an array too long to be worth materialising (vocabularies, merge tables)."""

    count: int
    item_type: int


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # row-major, i.e. GGUF dimensions reversed: a linear weight is (out, in)
    ggml_type: int
    offset: int  # from the start of its part's data section
    part: int

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def qtype(self) -> quant.QuantType:
        return quant.lookup(self.ggml_type)

    @property
    def nbytes(self) -> int:
        return self.qtype.nbytes(self.elements)


@dataclass
class GGUFFile:
    parts: list[Path]
    version: int
    kv: dict[str, Any]
    tensors: list[TensorInfo] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.parts[0]

    @property
    def architecture(self) -> str:
        return str(self.kv.get("general.architecture", ""))

    @property
    def file_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.parts)

    @property
    def tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def get(self, key: str, default: Any = None) -> Any:
        """Metadata lookup with the architecture prefix filled in: `get("block_count")`."""
        return self.kv.get(f"{self.architecture}.{key}", default)

    def count(self, key: str) -> int:
        """Length of an array-valued key, whether or not the array was materialised."""
        value = self.kv[key]
        return value.count if isinstance(value, ArraySummary) else len(value)


class _Reader:
    def __init__(self, f: BinaryIO, size: int, max_array_items: int) -> None:
        self.f, self.size, self.max_array_items = f, size, max_array_items

    def _need(self, n: int) -> None:
        if n < 0 or self.f.tell() + n > self.size:
            raise GGUFError(f"field of {n} bytes at offset {self.f.tell()} runs past the end of the file")

    def scalar(self, fmt: str) -> Any:
        n = struct.calcsize(fmt)
        self._need(n)
        return struct.unpack("<" + fmt, self.f.read(n))[0]

    def count(self, what: str) -> int:
        n = int(self.scalar("Q"))
        if n > MAX_COUNT:
            raise GGUFError(f"implausible {what} count {n}")
        return n

    def string(self) -> str:
        n = int(self.scalar("Q"))
        self._need(n)
        return self.f.read(n).decode("utf-8", "replace")

    def skip_string(self) -> None:
        n = int(self.scalar("Q"))
        self._need(n)
        self.f.seek(n, 1)

    def value(self, type_id: int) -> Any:
        if type_id in _SCALARS:
            return self.scalar(_SCALARS[type_id])
        if type_id == _STRING:
            return self.string()
        if type_id == _ARRAY:
            item_type, n = int(self.scalar("I")), self.count("array item")
            if item_type == _ARRAY:
                raise GGUFError("nested arrays are not supported")
            if n <= self.max_array_items:
                return [self.value(item_type) for _ in range(n)]
            if item_type in _SCALARS:
                nbytes = n * struct.calcsize(_SCALARS[item_type])
                self._need(nbytes)
                self.f.seek(nbytes, 1)
            elif item_type == _STRING:
                for _ in range(n):
                    self.skip_string()
            else:
                raise GGUFError(f"unknown array item type {item_type}")
            return ArraySummary(n, item_type)
        raise GGUFError(f"unknown metadata value type {type_id}")


def _read_part(
    path: Path, part: int, max_array_items: int, check_data: bool = True
) -> tuple[int, dict[str, Any], list[TensorInfo]]:
    size = path.stat().st_size
    with path.open("rb") as f:
        r = _Reader(f, size, max_array_items)
        if f.read(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        version = int(r.scalar("I"))
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.count("tensor"), r.count("metadata")
        kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(int(r.scalar("I")))
        tensors: list[TensorInfo] = []
        for _ in range(n_tensors):
            name, n_dims = r.string(), int(r.scalar("I"))
            if n_dims > 8:
                raise GGUFError(f"tensor {name!r} has {n_dims} dimensions")
            dims = tuple(int(r.scalar("Q")) for _ in range(n_dims))
            type_id, offset = int(r.scalar("I")), int(r.scalar("Q"))
            info = TensorInfo(name, tuple(reversed(dims)), type_id, offset, part)
            quant.lookup(type_id)  # raises on unknown types
            tensors.append(info)
        alignment = int(kv.get("general.alignment", 32))
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        for t in tensors if check_data else ():
            if data_start + t.offset + t.nbytes > size:
                raise GGUFError(f"tensor {t.name!r} extends past the end of {path.name}")
    return version, kv, tensors


def read(path: str | Path, max_array_items: int = 64, header_only: bool = False) -> GGUFFile:
    """Parse the header of a GGUF file. For a split model pass the first part; the rest are found next to it.

    Arrays longer than `max_array_items` are skipped and returned as `ArraySummary`.
    `header_only` accepts a file cut short after its tensor directory (see tools/fetch_gguf_header.sh):
    tensor data is not required to be present, and other parts of a split model are not read.
    """
    first = Path(path).expanduser()
    version, kv, tensors = _read_part(first, 0, max_array_items, check_data=not header_only)
    parts = [first]
    total = int(kv.get("split.count", 1))
    if total > 1 and not header_only:
        m = _SPLIT_RE.match(first.name)
        if not m or int(m.group(2)) != 1 or int(kv.get("split.no", 0)) != 0:
            raise GGUFError(f"{first.name} belongs to a {total}-part model; pass the first part (-00001-of-...)")
        for i in range(2, total + 1):
            part_path = first.with_name(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf")
            if not part_path.exists():
                raise GGUFError(f"missing part {part_path.name}")
            _, _, more = _read_part(part_path, i - 1, max_array_items)
            tensors.extend(more)
            parts.append(part_path)
    return GGUFFile(parts, version, kv, tensors)
