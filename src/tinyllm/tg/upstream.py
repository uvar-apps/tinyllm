"""Adapter over tinygrad's own LLM engine (`tinygrad.llm`).

This is the phase 0 and phase 1 engine, and the oracle every later engine is checked
against. It runs one batch of rows that all sit at the same position, which is what
upstream supports for Qwen3.8's recurrent layers.

Two output modes:
  sample: upstream's graph unchanged; sampling happens on the device and token ids come back.
  logits: the same graph up to the output head; full-vocabulary logits come back to the host.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer

from tinyllm.tg import (
    Buffer,
    BufferSpec,
    GlobalCounters,
    Tensor,
    TinyJit,
    UOp,
    default_device,
    dtypes,
    set_polling_wait,
    synchronize,
)
from tinyllm.tg import kernels as tl_kernels
from tinyllm.tg.sampler import OUTPUT, PARAMS, sample_row

# "0" sends every prefill chunk through the variable-length program, "1" gives full chunks their own fixed-length
# program; unset, the latter on AMD
STATIC_CHUNKS_ENV = "TINYLLM_STATIC_CHUNKS"


class _LogitsTransformer(Transformer):  # type: ignore[misc]
    """Upstream's Transformer with the in-graph sampler removed.

    `forward` mirrors `Transformer.forward` at the pinned commit up to the logits and must be
    re-checked on every tinygrad bump. tests/test_upstream.py asserts that argmax of these
    logits equals upstream's own greedy token.
    """

    def forward(self, tokens: Any, temperature: Any, *start_pos: Any) -> Any:
        x = self.token_embd(t := tokens.to(self.token_embd.weight.device)).float()
        if (hc := self.config.hc) is not None:
            x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
        elif self.config.attn_res_block_size:
            x = x.unsqueeze(2)
        idx_mask = None
        for block in self.blk:
            x, idx_mask = block(x, start_pos, idx_mask, t)
        if hc is not None:
            x = self.output_hc.mix(x)[0] if hasattr(self, "output_hc") else x.mean(2)
        elif self.config.attn_res_block_size:
            x = self.output_res_score.mix(x)
        return self.output(self.output_norm(x[:, -1:]))[:, -1, :].float().to(tokens.device)


def _mapped(n: int) -> tuple[Any, Any]:
    """A float32 tensor of n elements in GPU memory the CPU can map, and a NumPy view of it (no copy)."""
    buf = Buffer(default_device(), n * 4, options=BufferSpec(cpu_access=True)).ensure_allocated()
    view = buf._host_mv()
    if view is None:  # a device whose memory the host cannot map: an ordinary buffer would need copies
        raise RuntimeError(f"{default_device()} cannot map buffers into host memory")
    tensor = Tensor(UOp.from_buffer(buf, dtypes.float32))
    array = np.frombuffer(view, dtype=np.float32, count=n)
    array[:] = 0
    return tensor, array


@dataclass(frozen=True)
class StepRecord:
    """One call into the model: what ran, how long it took, how many bytes the kernels touched."""

    kind: str  # "prefill", "prefill_pipelined" or "decode"
    rows: int
    tokens: int  # tokens per row
    seconds: float
    bytes_touched: int
    calls: int = 1  # model calls covered; more than one only for "prefill_pipelined"


class UpstreamSession:
    """`rows` sequences advancing in lockstep through tinygrad.llm's Transformer."""

    def __init__(
        self,
        model_path: str | Path,
        max_context: int = 4096,
        rows: int = 1,
        chunk: int = 32,
        logits: bool = False,
        realize: bool | None = None,
        polling_wait: bool = True,
        kernels: bool | None = None,
        static_chunks: bool | None = None,
    ) -> None:
        self.rows, self.chunk, self.logits = rows, chunk, logits
        set_polling_wait(polling_wait)
        # tinyllm's own kernels (tinyllm.tg.kernels); TINYLLM_KERNELS selects them, "none" keeps tinygrad's paths
        if kernels is None or kernels:
            tl_kernels.install(tl_kernels.requested() if kernels is None else tl_kernels.KERNELS)
        self.kernels = tl_kernels.installed()
        start = time.perf_counter()
        # off the AMD kernel path upstream re-decodes the packed weights on every call unless they are realized
        if realize is None:
            realize = not default_device().startswith("AMD")
        self.model, self.kv = Transformer.from_gguf(str(model_path), max_context, realize=realize)
        if logits:
            self.model.__class__ = _LogitsTransformer
            self.model.prefill_jit = TinyJit(self.model.forward)
            self.model.rollout_jit = TinyJit(self.model.forward)
            self._sample_jit = TinyJit(self._sampled_forward)  # decode with the sampler in the program
        # full prefill chunks through a program compiled for exactly `chunk` tokens: with a symbolic token count
        # tinygrad gives elementwise kernels no workgroup, which makes the FFN gate several times slower
        if static_chunks is None:  # on by default where tinyllm's kernels run; the CPU only serves tests
            setting = os.environ.get(STATIC_CHUNKS_ENV)
            static_chunks = default_device().startswith("AMD") if setting is None else setting not in ("", "0")
        self._static_jit = TinyJit(self.model.forward) if static_chunks and chunk > 1 else None
        self.static_chunks = self._static_jit is not None
        self.load_seconds = time.perf_counter() - start
        self.max_context = int(self.model.max_context)
        self.vocab_size = int(self.model.config.vocab_size)
        self._v_pos = UOp.variable("start_pos", 0, self.max_context - 1)
        self._v_toks = UOp.variable("toks", 1, chunk)
        self._temp = Tensor([0.0])
        self._sampled: Any = None  # device tensor holding the ids sampled by the last call (sample mode)
        self._decode_in: Any = None  # logits mode: realized (rows, 1) input that each decode step overwrites
        self._logits_host: dict[int, Any] = {}  # logits mode: per output size, the host array outputs are copied into
        self._io: Any = None  # logits mode: `decode_sampled`'s mapped input and output (tensor, host view) pairs
        self._sampled_host: Any = None
        # logits mode: hand out that array itself, overwritten by the next call (for a caller that is done with the
        # logits by then, such as the engine); otherwise each call returns its own copy
        self.reuse_logits = False
        self.pos = 0
        self.trace: list[StepRecord] = []

    def tokenizer(self) -> Any:
        return SimpleTokenizer.from_gguf_kv(self.kv)

    def reset(self) -> None:
        """Start over at position 0. Upstream clears recurrent state itself when a call starts there."""
        self.pos, self._sampled = 0, None

    def _submit(self, tokens: Any, n_tokens: int) -> Any:
        """Queue one model call and return its output tensor without waiting for the device."""
        full = isinstance(tokens.shape[1], int) and tokens.shape[1] == self.chunk
        if self._static_jit is not None and full:
            # what Transformer.__call__ does, with this session's own program for full chunks
            if self.model._position_count is None:
                self.model._position_count = 1
            pos = UOp.variable("start_pos_0", 0, self.max_context - 1).bind(self.pos)
            out = self._static_jit(tokens.contiguous(), self._temp, pos).realize()
            self.pos += n_tokens
            return out
        out = self.model(tokens, (self._v_pos.bind(self.pos),), self._temp).realize()
        self.pos += n_tokens
        return out

    def _read(self, out: Any) -> Any:
        """Wait for `out` and bring it to the host."""
        if self.logits:
            # the output is a realized buffer: copy it straight into one reused host array (Tensor.numpy would
            # schedule a contiguous copy first and allocate a new megabyte for every token)
            buf = out.uop.base.buffer
            host = self._logits_host.get(buf.nbytes)
            if host is None:
                host = self._logits_host[buf.nbytes] = np.empty(tuple(int(d) for d in out.shape), dtype=np.float32)
            buf.ensure_allocated()
            Buffer("NPY", buf.nbytes, opaque=memoryview(host).cast("B")).copy_from(buf)
            result = host if self.reuse_logits else host.copy()
        else:
            self._sampled = out
            result = np.asarray(out.tolist(), dtype=np.int64).reshape(self.rows)
        synchronize()
        return result

    def _run(self, kind: str, tokens: Any, n_tokens: int) -> Any:
        GlobalCounters.reset()
        start = time.perf_counter()
        result = self._read(self._submit(tokens, n_tokens))
        self.trace.append(
            StepRecord(kind, self.rows, n_tokens, time.perf_counter() - start, int(GlobalCounters.global_mem))
        )
        return result

    def prefill(self, tokens: list[int], wait_each: bool = True) -> Any:
        """Feed `tokens` to every row, `chunk` tokens per call. Returns the output of the last call (see `step`).

        With `wait_each` every call is awaited and traced on its own, which is what a scheduler that
        interleaves other work needs. Without it the calls are queued back to back and only the last
        is awaited, as upstream's own loop does: host-side work for one chunk then overlaps the
        device's work on the previous one. That run is traced as one "prefill_pipelined" record.
        """
        end = self.pos + len(tokens)
        if not tokens:
            raise ValueError("prefill needs at least one token")
        if end >= self.max_context:
            raise ValueError(f"{end} tokens do not fit a context of {self.max_context}")
        buf = np.zeros((self.rows, self.max_context), dtype=np.int32)
        buf[:, self.pos : end] = np.asarray(tokens, dtype=np.int32)
        table = Tensor(buf)
        out, calls = None, 0
        GlobalCounters.reset()
        start = time.perf_counter()
        while self.pos < end:
            n = min(self.chunk, end - self.pos)
            lo = self._v_pos.bind(self.pos)
            # slicing by bound variables lets every chunk and every position reuse one compiled program;
            # with static chunks a full chunk has a fixed length and goes to its own program
            count = n if self._static_jit is not None and n == self.chunk else self._v_toks.bind(n)
            if wait_each:
                out = self._run("prefill", table[:, lo : lo + count], n)
            else:
                out, calls = self._submit(table[:, lo : lo + count], n), calls + 1
        if wait_each:
            return out
        result = self._read(out)
        seconds, touched = time.perf_counter() - start, int(GlobalCounters.global_mem)
        self.trace.append(StepRecord("prefill_pipelined", self.rows, len(tokens), seconds, touched, calls))
        return result

    def step(self, tokens: list[int] | Any | None = None) -> Any:
        """Advance every row by one token.

        `tokens` holds one id per row. In sample mode it may be omitted to continue with the ids
        the previous call sampled, which stay on the device as upstream's own loop keeps them.
        Returns ids of shape (rows,) in sample mode, logits of shape (rows, vocab) in logits mode.
        """
        if self.pos + 1 >= self.max_context:
            raise ValueError(f"context of {self.max_context} tokens is full")
        if tokens is None:
            if self._sampled is None:
                raise ValueError("no previously sampled tokens to continue from")
            return self._run("decode", self._sampled, 1)
        ids = [int(t) for t in tokens]
        if len(ids) != self.rows:
            raise ValueError(f"expected {self.rows} tokens, got {len(ids)}")
        if self.logits:
            return self._run_decode(ids)
        return self._run("decode", Tensor(ids, dtype="int32").reshape(self.rows, 1), 1)

    def _sampled_forward(self, io_in: Any, temperature: Any, io_out: Any, *start_pos: Any) -> Any:
        tokens = io_in[PARAMS].cast(dtypes.int32).reshape(1, 1)  # the token id travels with the settings
        packed = sample_row(self.model.forward(tokens, temperature, *start_pos), io_in[:PARAMS])
        return io_out.assign(packed).realize()

    def decode_sampled(self, ids: list[int], params: Any) -> Any:
        """A decode step that samples on the GPU (`tinyllm.tg.sampler`): (OUTPUT,) float32, logits mode, one row.

        Input and output live in GPU memory the CPU can map (`cpu_access`): the host writes the token and the
        settings in place and reads the result in place after the wait, so no copy runs through tinygrad's
        executor (each such copy costs about 0.5 ms of host time, more than replaying the decode program).
        The returned array is reused by the next call.
        """
        if not self.logits or self.rows != 1:
            raise ValueError("decode_sampled needs a one-row session in logits mode")
        if self.pos + 1 >= self.max_context:
            raise ValueError(f"context of {self.max_context} tokens is full")
        GlobalCounters.reset()
        start = time.perf_counter()
        if self._io is None:
            self._io = tuple(_mapped(n) for n in (PARAMS + 1, OUTPUT))
        (io_in, in_view), (io_out, out_view) = self._io
        in_view[:PARAMS] = np.asarray(params, dtype=np.float32)
        in_view[PARAMS] = float(ids[0])
        if self.model._position_count is None:
            self.model._position_count = 1
        self._sample_jit(io_in, self._temp, io_out, self._v_pos.bind(self.pos))
        self.pos += 1
        synchronize()
        result = self._sampled_host = out_view.copy() if self._sampled_host is None else self._sampled_host
        result[:] = out_view
        self.trace.append(
            StepRecord("decode", self.rows, 1, time.perf_counter() - start, int(GlobalCounters.global_mem))
        )
        return result

    def _write_decode_input(self, ids: list[int]) -> None:
        if self._decode_in is None:
            self._decode_in = Tensor.zeros(self.rows, 1, dtype="int32").contiguous().realize()
        host = np.asarray(ids, dtype=np.int32)
        self._decode_in.uop.buffer.copy_from(Buffer("NPY", host.nbytes, opaque=memoryview(host).cast("B")))
        if self.model._position_count is None:
            self.model._position_count = 1

    def _run_decode(self, ids: list[int]) -> Any:
        """A decode step with host-chosen tokens, keeping host work to the minimum (logits mode).

        `Transformer.__call__` builds a fresh input tensor each step, and realizing it runs tinygrad's scheduler
        over every live tensor: about 3 ms at full CPU clock, and 8.5 ms once the CPU drops to 1.5 GHz under a
        sustained GPU load on Strix Halo (docs/01-research-findings.md, section 5.6). Here the ids are copied into
        one realized buffer and the captured decode program is replayed directly.
        """
        GlobalCounters.reset()
        start = time.perf_counter()
        self._write_decode_input(ids)
        out = self.model.rollout_jit(self._decode_in, self._temp, self._v_pos.bind(self.pos))
        self.pos += 1
        result = self._read(out)
        self.trace.append(
            StepRecord("decode", self.rows, 1, time.perf_counter() - start, int(GlobalCounters.global_mem))
        )
        return result
