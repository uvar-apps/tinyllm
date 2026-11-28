"""A session's compiled programs, kept on disk between starts.

tinygrad builds a program (a `TinyJit`) by running its Python twice: the first call schedules and runs the graph
kernel by kernel, the second captures it and lowers it into command buffers. With every kernel's binary already in
tinygrad's compile cache a prefill program of Qwen3.8-Flash-Next still took 57 s and 59 s for those two calls,
all of it Python graph work, and a server start built more than a dozen such programs: over 20 minutes
(docs/04-qwen3.8-flash-next.md, section 16). `Programs` saves what the calls leave behind and puts it back at the
next start, which then runs neither.

What is saved, in one pickle after the warm-up (`save`):

- each captured program (tinygrad's `CapturedJit`: the compiled calls, before they are linked to addresses);
- the attributes the first calls set on the model's objects: the kernel chosen for each layer, weights re-viewed
  as packed bytes, caches and recurrent state. A later start has to find the objects as the programs left them,
  for the code outside the programs (state checkpoints) and for a program traced later.

What is not saved is the model. A buffer that existed when the session was constructed (`restore` walks the
session's objects then) is written as its place among them, "the first buffer of model.blk.3.attn_q.weight", and
bound to the buffer the next process loaded there. Of the buffers made later, those a program writes (attention
caches, recurrent state: 1.2 GB at 32,768 tokens) are written as their size and come back zeroed, as the model
creates them; the others (tables computed once, a program's constants) are written with their contents.

Objects that alias buffers from the host (the session itself, the n-gram row reader) are named in `unsaved`:
their attributes are left alone and they must create such buffers when they are constructed.

The file is valid for one `key`: the model files, the session's settings, tinygrad's and tinyllm's sources,
tinygrad's context variables and the device (`session_key`). It is pickled, so it is only read from a directory
of the user's own that nobody else can write.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import hashlib
import io
import json
import mmap
import os
import pickle
import re
import stat
import sys
import threading
import time
import types
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad import Tensor
from tinygrad.device import Buffer, BufferSpec
from tinygrad.engine import realize as tg_realize
from tinygrad.engine.jit import CapturedJit, _TinyJit
from tinygrad.helpers import ContextVar
from tinygrad.runtime.support.hcq2 import unwrap_lane
from tinygrad.uop.ops import Ops, ParamArg, UOp

FORMAT = 1
ENV = "TINYLLM_PROGRAM_CACHE"  # a directory to keep programs in; "0" builds them at every start
FILE, MANIFEST = "programs.pkl", "manifest.json"
KEPT = 4  # sessions' programs kept in the directory, the most recently used
# buffers written with their contents: more than this and something large is being copied that should have been
# found among the session's objects, so nothing is saved
MAX_VALUE_BYTES, MAX_VALUES_BYTES = 256 << 20, 1 << 30

Path_ = tuple[tuple[str, Any], ...]  # the way to an object from the session: ("a", attribute) and ("i", item) steps


class Refused(Exception):
    """The programs cannot be saved or loaded as they are; the session builds them as before."""


def directory(environ: Mapping[str, str] = os.environ) -> Path | None:
    """Where programs are kept: TINYLLM_PROGRAM_CACHE, or `tinyllm/programs` in the user's cache directory."""
    setting = environ.get(ENV)
    if setting is not None:
        return None if setting.strip().lower() in ("", "0", "off", "none") else Path(setting).expanduser()
    return Path(environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "tinyllm" / "programs"


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def model_files(path: Path) -> list[Path]:
    """`path`, or every part of the split model it is the first part of."""
    if (m := re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", path.name)) is None:
        return [path]
    return sorted(path.parent.glob(f"{glob_escape(m.group(1))}-?????-of-{m.group(2)}.gguf"))


def glob_escape(name: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", name)


def session_key(files: Sequence[Path], settings: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """The key programs are valid for, and what it is made of (kept beside them for a person to read)."""
    import tinygrad

    from tinyllm.tg import device_info

    environment = {
        k: v
        for k, v in sorted(os.environ.items())
        if k != ENV and (k.startswith(("TINYLLM_", "BEAM", "JITBEAM", "AMD", "HCQ", "KFD")) or k in ("DEV", "HALF"))
    }
    parts: dict[str, Any] = {
        "format": FORMAT,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "tinygrad": _tree_hash(Path(tinygrad.__file__).parent),
        "tinyllm": _tree_hash(Path(__file__).parent),
        "device": device_info(),
        "context": {k: str(v.value) for k, v in sorted(ContextVar._cache.items())},
        "environment": environment,
        "files": [[p.name, (s := p.stat()).st_size, s.st_mtime_ns] for p in files],
        "settings": {k: settings[k] for k in sorted(settings)},
    }
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest(), parts


# ---- which buffers the programs write ----

_written: weakref.WeakSet[UOp] = weakref.WeakSet()
_recorder_installed = False


def _install_recorder() -> None:
    """Note the buffers each program writes as tinygrad lowers it: `hcq_compile` receives a JIT's calls as compiled
    kernels with their output arguments known, before they are merged into command buffers."""
    global _recorder_installed
    if _recorder_installed:
        return
    original = tg_realize.hcq_compile

    def hcq_compile(linear: UOp, input_uops: Any, profile: bool, cache: bool = False) -> UOp:
        if input_uops is not None and not cache:  # a JIT's program (run_linear caches, beam timing has no inputs)
            for call in linear.src:
                call = call.without_after
                if call.op is not Ops.CALL:
                    continue
                for k in tg_realize.get_call_outs_ins(call)[0]:
                    base = unwrap_lane(call.src[1 + k])[0]
                    if base.op is Ops.BUFFER:
                        _written.add(base)
        return original(linear, input_uops, profile, cache=cache)

    tg_realize.hcq_compile = hcq_compile
    _recorder_installed = True


# ---- the session's objects ----

_MISSING = object()
_LEAVES: tuple[type, ...] = (
    str, bytes, bytearray, memoryview, int, float, complex, type(None), type, range, set, frozenset, enum.Enum,
    np.ndarray, np.generic, UOp, Buffer, _TinyJit, CapturedJit, mmap.mmap, threading.Thread, Executor, io.IOBase,
    types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, functools.partial,
)  # fmt: skip


def _attributes(obj: Any) -> dict[str, Any]:
    out = dict(vars(obj)) if hasattr(obj, "__dict__") else {}
    for klass in type(obj).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if not name.startswith("__") and (value := getattr(obj, name, _MISSING)) is not _MISSING:
                out[name] = value
    return out


def _buffers(uop: UOp) -> list[UOp]:
    return [u for u in uop.toposort() if u.op is Ops.BUFFER and u.arg.buffer is not None]


@dataclass
class _Snapshot:
    """The objects reachable from a session, as they are when it has been constructed."""

    objects: dict[int, tuple[Path_, Any, dict[str, Any]]] = field(
        default_factory=dict
    )  # by id: path, object, attributes
    tensors: dict[int, tuple[Path_, Tensor, UOp, Any]] = field(default_factory=dict)  # by id: path, tensor, uop, owner
    buffers: dict[UOp, tuple[Path_, int]] = field(default_factory=dict)  # a tensor's path and the buffer's place in it

    def walk(self, obj: Any, path: Path_, owner: Any, seen: set[int]) -> None:
        if isinstance(obj, Tensor):
            if id(obj) not in self.tensors:
                self.tensors[id(obj)] = (path, obj, obj.uop, owner)
                for i, u in enumerate(_buffers(obj.uop)):
                    self.buffers.setdefault(u, (path, i))
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (list, tuple)):
            for i, x in enumerate(obj):
                if not isinstance(x, _LEAVES):
                    self.walk(x, (*path, ("i", i)), owner, seen)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, (str, int)) and not isinstance(v, _LEAVES):
                    self.walk(v, (*path, ("i", k)), owner, seen)
        elif not isinstance(obj, Programs):
            attributes = _attributes(obj)
            self.objects[id(obj)] = (path, obj, attributes)
            for name, value in attributes.items():
                if not isinstance(value, _LEAVES):
                    self.walk(value, (*path, ("a", name)), obj, seen)

    @classmethod
    def of(cls, root: Any) -> _Snapshot:
        snapshot = cls()
        snapshot.walk(root, (), None, set())
        return snapshot


def _tensor(uop: UOp) -> Tensor:
    return Tensor(uop)  # through the constructor: tinygrad keeps track of every live Tensor


def _plain(u: UOp) -> bool:
    """Whether a BUFFER node is what `UOp.new_buffer` makes: its size, type and device say all there is to it."""
    arg, buf = u.arg, u.arg.buffer
    return (
        isinstance(buf, Buffer)
        and buf._base is None
        and buf.options == BufferSpec()
        and arg == ParamArg(arg.slot, arg.dtype, device=arg.device, buffer=buf)
        and UOp._buffer(buf.nbytes // arg.dtype.itemsize, arg.dtype, arg.slot, device=arg.device, buffer=buf) is u
    )


class _Pickler(pickle.Pickler):
    def __init__(self, file: Any, snapshot: _Snapshot) -> None:
        super().__init__(file, protocol=5)
        self.snapshot = snapshot
        self.zeroed: dict[UOp, int] = {}  # buffers the programs write, made after the session was constructed
        self.values: dict[UOp, int] = {}  # buffers written with their contents, and their bytes
        self.held: set[UOp] = set()  # buffers of the session as constructed

    def persistent_id(self, obj: Any) -> Any:
        kind = type(obj)
        if kind is UOp:
            if obj.op is not Ops.BUFFER or obj.arg.buffer is None:
                return None
            if (place := self.snapshot.buffers.get(obj)) is not None:
                self.held.add(obj)
                return ("held", place, int(obj.arg.buffer.nbytes), obj.arg.dtype, obj.arg.device)
            buf = obj.arg.buffer
            if obj in _written and _plain(obj):
                index = self.zeroed.setdefault(obj, len(self.zeroed))
                return ("zeros", index, obj.arg.device, int(buf.nbytes) // obj.arg.dtype.itemsize, obj.arg.dtype)
            if obj not in self.values:
                nbytes = int(buf.nbytes) if buf.is_allocated() else 0
                if not isinstance(buf, Buffer) or str(obj.arg.device).startswith("DISK") or nbytes > MAX_VALUE_BYTES:
                    raise Refused(f"a buffer of {nbytes} bytes on {obj.arg.device} is held by no object of the session")
                self.values[obj] = nbytes
                if sum(self.values.values()) > MAX_VALUES_BYTES:
                    raise Refused(f"over {MAX_VALUES_BYTES >> 20} MB of buffers would be written with their contents")
            return None
        if kind in (int, str, tuple, float, bytes, bool, type(None)):
            return None
        if (tensor := self.snapshot.tensors.get(id(obj))) is not None and tensor[1] is obj:
            return ("tensor", tensor[0])
        if (known := self.snapshot.objects.get(id(obj))) is not None and known[1] is obj:
            return ("object", known[0])
        return None

    def reducer_override(self, obj: Any) -> Any:
        if type(obj) is Tensor:
            return _tensor, (obj.uop,)
        return NotImplemented


# what a saved file may name: tinygrad's and tinyllm's classes and functions, and the few of Python's own that
# their pickles use
_GLOBALS = {
    ("builtins", n) for n in ("tuple", "list", "dict", "set", "frozenset", "bytes", "bytearray", "slice", "range",
                              "int", "float", "complex", "bool", "str", "getattr")
} | {("collections", "OrderedDict"), ("collections", "defaultdict"), ("collections", "deque"),
     ("copyreg", "_reconstructor"), ("copyreg", "__newobj__"), ("_codecs", "encode"), ("decimal", "Decimal"),
     ("pathlib", "PosixPath"), ("pathlib._local", "PosixPath"), ("functools", "partial")}  # fmt: skip


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: Any, snapshot: _Snapshot) -> None:
        super().__init__(file)
        self.buffers = {place: u for u, place in snapshot.buffers.items()}
        self.tensors = {path: t for path, t, _, _ in snapshot.tensors.values()}
        self.objects = {path: obj for path, obj, _ in snapshot.objects.values()}
        self.zeroed: dict[int, UOp] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] in ("tinygrad", "tinyllm", "numpy") or (module, name) in _GLOBALS:
            return super().find_class(module, name)
        raise Refused(f"the file names {module}.{name}, which programs have no use for")

    def persistent_load(self, pid: Any) -> Any:
        kind = pid[0]
        if kind == "held":
            _, place, nbytes, dtype, device = pid
            u = self.buffers.get(place)
            if u is None or int(u.arg.buffer.nbytes) != nbytes or u.arg.dtype != dtype or u.arg.device != device:
                raise Refused(f"the session has no {nbytes}-byte {dtype} buffer at {_show(place[0])}")
            return u
        if kind == "zeros":
            _, index, device, numel, dtype = pid
            if (u := self.zeroed.get(index)) is None:
                u = self.zeroed[index] = UOp.new_buffer(device, numel, dtype)
            return u
        found = (self.tensors if kind == "tensor" else self.objects if kind == "object" else {}).get(pid[1], _MISSING)
        if found is _MISSING:
            raise Refused(f"the session has no {kind} at {_show(pid[1])}")
        return found


def _show(path: Path_) -> str:
    return "session" + "".join(f".{step}" if kind == "a" else f"[{step!r}]" for kind, step in path)


def _private(path: Path) -> bool:
    """Whether only this user can have written `path`."""
    info = path.stat()
    return info.st_uid == os.getuid() and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


class Program(_TinyJit):  # type: ignore[misc]
    """A `TinyJit` that a session may capture on its first call.

    tinygrad runs a program's Python twice before it replays it: the first call runs the graph as it is, the
    second captures it. The first is there for what a function does once: weights it realizes, caches it makes.
    Captured, those would run at every call. A program whose function another program has already been through
    (`like`: the same function at another length) has nothing left to do once, so its first call can capture, and
    the first call's half of the build is saved (57 of 116 s for a prefill program of Qwen3.8-Flash-Next).

    The check: a capture records one linear per realize. If this program's first call recorded another number
    than `like`'s capture did, it did something on the way after all; that is done now, and the next call
    captures again, as it would have after an ordinary first call."""

    cnt: int  # tinygrad's: calls so far (0: run as it is, 1: capture, then replay)
    captured: Any
    like: Program | None = None
    linears = 0  # realizes the capture recorded
    early = False  # captured by its first call
    again = False  # captured by its first call, then once more: the first call had made something

    def add_linear(self, linear: UOp, var_vals: dict[str, int]) -> None:
        self.linears += 1
        super().add_linear(linear, var_vals)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        like = self.like
        early = self.cnt == 0 and like is not None and like.captured is not None and like.linears > 0
        if early:
            self.cnt = 1
        if self.cnt == 1:
            self.linears = 0
        out = super().__call__(*args, **kwargs)
        if early:
            assert like is not None
            self.early = self.linears == like.linears
            if not self.early:
                self.captured = None
                self.cnt, self.again = 1, True
        return out


class Programs:
    """The programs of one session: made by `jit`, put back by `restore` when a file for `key` exists, written by
    `save`. `directory` None: programs are built at every start, as without this class."""

    def __init__(self, root: Any, directory: Path | None = None, unsaved: tuple[type, ...] = ()) -> None:
        self.root, self.directory, self.unsaved = root, directory, unsaved
        self.jits: dict[str, Any] = {}
        self.key, self.label = "", "session"
        self.parts: dict[str, Any] = {}
        self.restored = False  # `restore` put the saved programs and attributes back
        self.loaded: list[str] = []  # the programs among them this session has
        self.note = ""  # why nothing was loaded or saved, for the log
        self.seconds = 0.0  # spent loading or saving
        self.bytes = 0
        self._snapshot: _Snapshot | None = None
        if directory is not None:
            _install_recorder()

    def jit(self, name: str, fn: Any, like: Any = None) -> Any:
        """A program of `fn`, saved and restored under `name`. `like`: a program of the same function (at another
        length, say); once that one is captured, this one is captured by its first call (`Program`)."""
        if name in self.jits:
            raise ValueError(f"two programs named {name}")
        self.jits[name] = jit = Program(fn)
        jit.like = like
        return jit

    @property
    def path(self) -> Path | None:
        return None if self.directory is None or not self.key else self.directory / f"{self.label}-{self.key[:16]}"

    def identify(self, label: str, files: Sequence[Path], settings: Mapping[str, Any]) -> None:
        """Name what the programs are valid for (`session_key`); `label` starts their directory's name."""
        if self.directory is not None:
            self.label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)[:80]
            self.key, self.parts = session_key(files, settings)

    def restore(self) -> bool:
        """Note the session's objects as constructed, then put back the saved programs and the attributes their
        first calls set, if a file for this session exists. False: the session builds its programs."""
        if (path := self.path) is None:
            return False
        start = time.perf_counter()
        snapshot = self._snapshot = _Snapshot.of(self.root)
        file = path / FILE
        if not file.exists():
            self.note = "not loaded: none saved for this model, these settings and these sources"
            return False
        try:
            if not all(_private(p) for p in (path.parent, path, file)):
                raise Refused(f"{file} could have been written by someone else")
            with open(file, "rb") as f:
                unpickler = _Unpickler(f, snapshot)
                payload = unpickler.load()
            if payload.get("format") != FORMAT or payload.get("key") != self.key:
                raise Refused("the file is for another session")
            for u in unpickler.zeroed.values():  # as the model creates them: zeroed
                buf = u.buffer.ensure_allocated()
                if buf.nbytes:
                    buf.copy_from(Buffer("NPY", buf.nbytes, opaque=memoryview(np.zeros(buf.nbytes, dtype=np.uint8))))
        except (Refused, pickle.UnpicklingError, EOFError, OSError, AttributeError, ImportError, LookupError,
                TypeError, ValueError, RuntimeError, AssertionError) as e:  # fmt: skip
            self.note = f"not loaded: {e}" if isinstance(e, Refused) else f"not loaded: {type(e).__name__}: {e}"
            return False
        # nothing above touched the session; from here on it is as the saving process left it
        for tensor_path, uop in payload["tensors"]:
            unpickler.tensors[tensor_path].uop = uop
        for object_path, name, value in payload["attributes"]:
            setattr(unpickler.objects[object_path], name, value)
        for name, captured in payload["programs"].items():
            if (jit := self.jits.get(name)) is not None and jit.captured is None:
                jit.captured, jit.cnt = captured, 2
                self.loaded.append(name)
        with contextlib.suppress(OSError):
            os.utime(path)  # the most recently used are kept
        self.restored, self.seconds, self.bytes = True, time.perf_counter() - start, file.stat().st_size
        self.note, self._snapshot = "", None
        return True

    def pending(self) -> list[str]:
        """The programs that have run once: their next call captures them."""
        return [name for name, jit in self.jits.items() if jit.cnt == 1]

    def save(self) -> bool:
        """Write the captured programs and what their first calls set on the session's objects. False (with
        `note`) when there is nowhere to write, they were loaded from there, or they cannot be saved."""
        if (path := self.path) is None or self._snapshot is None or self.restored:
            return False
        start = time.perf_counter()
        snapshot, self._snapshot = self._snapshot, None
        programs = {name: jit.captured for name, jit in self.jits.items() if jit.captured is not None}
        attributes, tensors = [], []
        for object_path, obj, before in snapshot.objects.values():
            if isinstance(obj, self.unsaved):
                continue
            for name, value in _attributes(obj).items():
                if name not in before or before[name] is not value:
                    attributes.append((object_path, name, value))
        for tensor_path, tensor, uop, owner in snapshot.tensors.values():
            if tensor.uop is not uop and not isinstance(owner, self.unsaved):
                tensors.append((tensor_path, tensor.uop))
        payload = {
            "format": FORMAT,
            "key": self.key,
            "attributes": attributes,
            "tensors": tensors,
            "programs": programs,
        }
        temporary = path / f"{FILE}.{os.getpid()}"
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
                pickler = _Pickler(f, snapshot)
                pickler.dump(payload)
            os.replace(temporary, path / FILE)
        except (Refused, pickle.PicklingError, TypeError, AttributeError, RecursionError, OSError) as e:
            self.note = f"not saved: {e}" if isinstance(e, Refused) else f"not saved: {type(e).__name__}: {e}"
            with contextlib.suppress(OSError):
                temporary.unlink()
            with contextlib.suppress(OSError):
                path.rmdir()  # when this left it empty
            return False
        self.seconds, self.bytes, self.note = time.perf_counter() - start, (path / FILE).stat().st_size, ""
        manifest = {
            "key": self.key,
            "programs": sorted(programs),
            "attributes": len(attributes),
            "tensors": len(tensors),
            "buffers": {
                "of the session": [len(pickler.held), sum(int(u.arg.buffer.nbytes) for u in pickler.held)],
                "zeroed at load": [len(pickler.zeroed), sum(int(u.arg.buffer.nbytes) for u in pickler.zeroed)],
                "with their contents": [len(pickler.values), sum(pickler.values.values())],
            },
            "bytes": self.bytes,
            "valid for": self.parts,
        }
        with contextlib.suppress(OSError):  # for a person to read; nothing loads it
            (path / MANIFEST).write_text(json.dumps(manifest, indent=1, default=str) + "\n")
        self._prune(path)
        return True

    def _prune(self, keep: Path) -> None:
        """Drop all but the most recently used sessions' programs: every change of settings or sources makes a
        new directory."""
        assert self.directory is not None
        others = [p for p in self.directory.iterdir() if p.is_dir() and p != keep and (p / FILE).exists()]
        for old in sorted(others, key=lambda p: p.stat().st_mtime, reverse=True)[KEPT - 1 :]:
            with contextlib.suppress(OSError):
                for file in old.iterdir():
                    file.unlink()
                old.rmdir()
