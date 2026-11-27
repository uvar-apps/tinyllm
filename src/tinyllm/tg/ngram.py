"""Qwen3.8-Flash-Next's n-gram embedding rows, read from the GGUF file on the host.

At one block the model adds an embedding of the bigrams and trigrams ending at each token: 16 rows of 160 IQ4_NL
weights (90 bytes each) per token, picked by hashing from a table of 320 million rows (28.8 GB) that tinygrad
leaves in the file (`tinygrad.llm.gguf.GGUFPackedTable`). Before each call the rows of its positions are copied
into a host buffer, and the call copies them to the GPU and decodes them (`PLELayer.write_rows` and `__call__`).

tinygrad copies the rows out of its own mapping of the file. That mapping asks for transparent huge pages
(`tinygrad.runtime.ops_disk.DiskDevice`), and a fault on a row that is not in the page cache then reads a large
block of the file (btrfs, kernel 7.2). One token's 16 rows took 63 ms that way, twice a whole decode step
(docs/04-qwen3.8-flash-next.md). `FileRows` reads them from its own descriptor with readahead off
(POSIX_FADV_RANDOM). It first tries a non-blocking read of every row (RWF_NOWAIT), which copies a row that is in
the page cache in under a microsecond and refuses one that is not. The rows refused are then read from a pool of
threads, so the NVMe serves them in parallel: 0.4 ms for a token's 16 rows when none are cached. On a file system
without non-blocking reads (tmpfs) every row goes to the pool.

Calls of many positions (prefill) gather their rows with numpy from a mapping of the table instead, split over the
pool (`MAPPED`): 5 ms for the 32,768 rows of a 2,048-token call when they are cached. Calls of a few positions
(decoding, a speculative verify) must not take that path: its faults read one row after the other, and between GPU
calls a fault took 0.8 to 2.7 ms, which made a decoded token up to 81 ms and a four-row speculative cycle up to
250 ms (docs/04-qwen3.8-flash-next.md, section 15). Through the pool the same rows take 2 to 4 ms a cycle.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import itertools
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import mv_address

RWF_NOWAIT = 0x8  # linux/fs.h; this Python's os.preadv does not take flags
THREADS = 16
# threads a mapped write of many positions is split over: they wait on page faults, and the NVMe serves that many
# reads at once (a prompt's new text is rarely in the page cache)
GATHER_THREADS = 64
PAGE = 4096
MAPPED = True  # calls of MAPPED_SPLIT positions or more gather their rows from a mapping of the file with numpy
MAPPED_SPLIT = 64  # fewer positions (decoding, a speculative verify) read each row on its own
PREFETCH = True  # the next prefill call's rows written by a background thread while the current call runs


def row_ids(ple: Any, tokens: list[int], start: int, n: int) -> np.ndarray:
    """(n, heads) table rows of positions start .. start + n - 1, as `PLELayer.write_rows` (tinygrad.llm.model,
    pinned commit) hashes them: the n-grams ending at each token, EOS before the start or an EOS. In numpy: the
    per-position Python lists of `write_rows` cost a second a call inside a model process (the collector walks its
    millions of objects). Exact: a token id times a multiplier stays below 2**63."""
    k, heads = len(ple.multipliers), len(ple.vocab_sizes)
    lo = max(0, start - (k - 1))
    t = np.asarray(tokens[lo : start + n], dtype=np.int64)
    p = np.arange(start, start + n)
    at = p - lo  # position p in t
    is_eos = t == ple.eos
    acc = t[at] * ple.multipliers[0]
    blocked, hashes = np.zeros(n, dtype=bool), []
    for i in range(1, k):  # the token i before p, or EOS if there is none or an EOS lies between them
        valid = p >= i
        j = np.where(valid, at - i, 0)
        blocked = blocked | ~valid | (valid & is_eos[j])
        acc = acc ^ (np.where(blocked, ple.eos, t[j]) * ple.multipliers[i])
        hashes.append(acc)
    picked = np.stack([hashes[j * (k - 1) // heads] for j in range(heads)], axis=1)
    offsets = np.array(list(itertools.accumulate(ple.vocab_sizes, initial=0))[:-1], dtype=np.int64)
    return picked % np.array(ple.vocab_sizes, dtype=np.int64) + offsets


class _IOVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_preadv2 = _libc.preadv2
_preadv2.argtypes = [ctypes.c_int, ctypes.POINTER(_IOVec), ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
_preadv2.restype = ctypes.c_ssize_t


class FileRows:
    """Stands in for a `GGUFPackedTable`: `write` reads the rows from the file itself, `read` is the table's own."""

    def __init__(self, table: Any) -> None:
        self.inner = table
        device = str(table.data.device)
        if not device.startswith("DISK:"):
            raise ValueError(f"the n-gram table is not read from a file (device {device})")
        offset = table.data.uop.contiguous_view_offset()
        if offset is None:
            raise ValueError("the n-gram table is not one contiguous range of its file")
        self.path, self.base, self.row_bytes = device[len("DISK:") :], int(offset), int(table.row_bytes)
        self.fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.pool = ThreadPoolExecutor(max(THREADS, GATHER_THREADS), thread_name_prefix="ngram-rows")
        self.table_bytes = int(table.data.numel()) * table.data.dtype.itemsize
        self._map: mmap.mmap | None = None  # the table's range of the file (`_rows`), made on first use
        self._rows: np.ndarray | None = None  # the table as (rows, row bytes)
        self._slots_array: np.ndarray | None = None  # the host slots as (slots, rows per position, row bytes)
        self._view: memoryview | None = None
        self._address = 0
        self._nowait = True  # until the file system refuses RWF_NOWAIT
        self.cold = self.warm = 0  # rows read by the pool and by the non-blocking pass

    def _slots(self, per_position: int, slots: int) -> None:
        # the host buffer tinygrad's table reads from, made as `GGUFPackedTable.write` makes it
        table = self.inner
        table.rows = Tensor.empty(slots, per_position, self.row_bytes, dtype=dtypes.uint8, device="CPU").realize()
        self._view = table.rows._buffer().as_memoryview(allow_zero_copy=True)
        self._address = mv_address(self._view)
        self._slots_array = np.frombuffer(self._view, dtype=np.uint8).reshape(slots, per_position, self.row_bytes)

    def _read(self, iov: _IOVec, dst: int, src: int, flags: int) -> bool:
        iov.base, iov.len = dst, self.row_bytes
        n = _preadv2(self.fd, ctypes.byref(iov), 1, src, flags)
        if n == self.row_bytes:
            return True
        if n < 0 and (err := ctypes.get_errno()) not in (errno.EAGAIN, errno.EINTR):
            if err == errno.EOPNOTSUPP and flags & RWF_NOWAIT:
                self._nowait = False
                return False
            raise OSError(err, f"reading n-gram rows from {self.path}: {os.strerror(err)}")
        return False

    def _read_blocking(self, rows: list[tuple[int, int]]) -> None:
        iov = _IOVec()
        for dst, src in rows:
            while not self._read(iov, dst, src, 0):  # a short read or an interrupted one: read the row again
                pass

    def _mapped(self) -> np.ndarray:
        """The table as (rows, row bytes), viewed through a mapping of its range of the file. The mapping asks
        for no readahead (MADV_RANDOM) and none of tinygrad's huge pages."""
        if self._rows is None:
            first = self.base // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
            self._map = mmap.mmap(self.fd, self.base - first + self.table_bytes, prot=mmap.PROT_READ, offset=first)
            self._map.madvise(mmap.MADV_RANDOM)
            self._map_first = first
            rows = np.frombuffer(self._map, dtype=np.uint8, count=self.table_bytes, offset=self.base - first)
            self._rows = rows.reshape(-1, self.row_bytes)
        return self._rows

    def _write_mapped(self, start: int, ids: list[list[int]] | np.ndarray) -> None:
        """`write` through the mapping: numpy gathers the rows (a page not in the page cache is read by the fault
        that touches it). Many positions are split over the thread pool: numpy releases the interpreter lock while
        it copies, so up to GATHER_THREADS faults read from the NVMe at once."""
        table, slots = self._mapped(), self._slots_array
        assert slots is not None
        rows = np.asarray(ids, dtype=np.int64)
        self.warm += rows.size
        if len(rows) < THREADS:
            slots[start : start + len(rows)] = table[rows]
            return
        parts = min(GATHER_THREADS, len(rows))
        bounds = np.linspace(0, len(rows), parts + 1).astype(int)

        def part(i: int) -> None:
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            slots[start + lo : start + hi] = table[rows[lo:hi]]

        list(self.pool.map(part, range(parts)))

    def write(self, start: int, ids: list[list[int]] | np.ndarray, slots: int) -> None:
        """`GGUFPackedTable.write`: the rows `ids[k]` of position start + k into the host slots."""
        if self._view is None:
            self._slots(len(ids[0]), slots)
        if MAPPED and len(ids) >= MAPPED_SPLIT:
            return self._write_mapped(start, ids)
        per, rb = len(ids[0]), self.row_bytes
        iov, cold = _IOVec(), []
        dst = self._address + start * per * rb
        for row in np.asarray(ids, dtype=np.int64).reshape(-1).tolist():
            src = self.base + row * rb
            if not (self._nowait and self._read(iov, dst, src, RWF_NOWAIT)):
                cold.append((dst, src))
            dst += rb
        self.warm += len(ids) * per - len(cold)
        if cold:
            self.cold += len(cold)
            list(self.pool.map(self._read_blocking, (cold[i::THREADS] for i in range(min(THREADS, len(cold))))))

    def zero(self, start: int, n: int) -> None:
        """Zero the host slots of positions start .. start + n - 1: the rows a padded call reads for its padding
        (zeros decode to zeros, where a slot never written could hold anything)."""
        assert self._slots_array is not None, "the host slots are made by the first write, or by `install`"
        self._slots_array[start : start + n] = 0

    def read(self, start: Any, n: int, device: Any) -> Tensor:
        return self.inner.read(start, n, device)


def install(ple: Any, slots: int | None = None) -> FileRows:
    """Route a `PLELayer`'s row copies through `FileRows` (once). With `slots`, the positions a call may write
    rows for, the host buffer is made now instead of by the first `write`: a session whose programs are loaded from
    disk (tinyllm.tg.programs) binds them to the buffers that exist when it has been constructed."""
    if not isinstance(ple.table, FileRows):
        ple.table = FileRows(ple.table)
    if slots is not None and ple.table._view is None:
        ple.table._slots(len(ple.ple.vocab_sizes), slots)
    return ple.table
