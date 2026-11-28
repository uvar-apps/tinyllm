"""Lowering a program's kernels in tinygrad's worker processes when its JIT tunes them.

tinygrad lowers kernels (rewrites, linearizes, renders and compiles each) in a pool of worker processes, except
when any kernel of the batch is to be tuned by BEAM search: the search times its candidates on the device, which
only the main process may open, so then the whole batch is lowered there, one kernel after another
(`tinygrad.engine.realize.lower_and_compile`). A JIT with JITBEAM set marks every generic kernel that way, so a
prefill program of Qwen3.8-Flash-Next lowered its 105 tuned kernels for about 30 s in the main process at every
build, although the searches' results had been on disk since the first one
(docs/04-qwen3.8-flash-next.md, section 16).

A search whose result is on disk never touches the device. `install` puts a step before tinygrad's own: every
kernel of the batch not lowered yet goes to the pool, where a worker lowers it unless its search is not on disk;
tinygrad then finds the others in its cache and lowers what is left, the searches that have to run, in the main
process as before. The programs are the same either way: a worker applies the options the search stored.
"""

from __future__ import annotations

from typing import Any

from tinygrad.codegen import to_program, to_program_cache, to_program_context, to_program_key
from tinygrad.engine import realize as tg_realize
from tinygrad.engine.worker import get_worker_pool
from tinygrad.helpers import CACHELEVEL, IGNORE_BEAM_CACHE, Context, diskcache_get
from tinygrad.uop.ops import Ops, UOp

MIN_KERNELS = 4  # fewer are not worth the round trip to the pool


class _NotOnDisk(Exception):
    """A kernel's BEAM search has no stored result: it has to run, in the main process."""


_guarded = False


def _guard_search() -> None:
    """In a worker: make a BEAM search that would have to run raise instead (it would open the device)."""
    from tinygrad.codegen.opt import search

    original = search.beam_search

    def beam_search(s: Any, amt: int, allow_test_size: bool = True) -> Any:
        # the key `tinygrad.codegen.opt.search.beam_search` looks its result up by, at the pinned commit
        key = {"ast": s.ast.key, "amt": amt, "allow_test_size": allow_test_size}
        key |= {"device": s.ren.target.device, "suffix": s.ren.suffix}
        if IGNORE_BEAM_CACHE or CACHELEVEL < 1 or diskcache_get("beam_search", key) is None:
            raise _NotOnDisk
        return original(s, amt, allow_test_size)

    search.beam_search = beam_search


def _lower(task: tuple[int, UOp, Any, dict[str, Any]]) -> tuple[int, UOp | None]:
    """In a worker: the compiled program of one kernel, or None when its search is not on disk."""
    global _guarded
    index, ast, renderer, context = task
    if not _guarded:
        _guard_search()
        _guarded = True
    try:
        with Context(**context):
            return index, to_program(ast, renderer)
    except _NotOnDisk:
        return index, None


def lower_in_pool(linear: UOp) -> int:
    """Lower the kernels of `linear` that are not lowered yet in the worker pool, if any of them is to be tuned
    (otherwise tinygrad sends them there itself). Returns how many the pool lowered."""
    calls = {}
    for c in linear.toposort():
        if c.op is Ops.CALL and (found := tg_realize._get_call_to_compile(c)) is not None:
            calls[c] = found
    if not any(getattr(c.body.arg, "beam", 0) for c in calls):
        return 0
    todo = {key: found for found in calls.values() if (key := to_program_key(*found)) not in to_program_cache}
    if len(todo) < MIN_KERNELS or (pool := get_worker_pool()) is None:
        return 0
    context = {v.key: v.value for v in to_program_context}
    keys = list(todo)
    done = 0
    try:
        for index, program in pool.imap_unordered(_lower, [(i, *todo[key], context) for i, key in enumerate(keys)]):
            if program is not None:
                to_program_cache[keys[index]] = program
                done += 1
    except Exception:  # whatever a worker could not do, tinygrad's own lowering does next, or reports
        pass
    return done


def install() -> None:
    """Put `lower_in_pool` before tinygrad's lowering of a batch of kernels (once per process)."""
    original = tg_realize.lower_and_compile
    if getattr(original, "_tinyllm_pool", False):
        return

    def lower_and_compile(linear: UOp, verbose: bool = True) -> UOp:
        lower_in_pool(linear)
        return original(linear, verbose)

    lower_and_compile._tinyllm_pool = True  # type: ignore[attr-defined]
    tg_realize.lower_and_compile = lower_and_compile
