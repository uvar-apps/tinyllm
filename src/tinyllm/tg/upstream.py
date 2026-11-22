"""Adapter over tinygrad's own LLM engine (`tinygrad.llm`).

This is the phase 0 and phase 1 engine, and the oracle every later engine is checked
against. It runs one batch of rows that all sit at the same position, which is what
upstream supports for Qwen3.8's recurrent layers.

Two output modes:
  sample: upstream's graph unchanged; sampling happens on the device and token ids come back.
  logits: the same graph up to the output head; full-vocabulary logits come back to the host.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer

from tinyllm.tg import GlobalCounters, Tensor, TinyJit, UOp, default_device, set_polling_wait, synchronize


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
    ) -> None:
        self.rows, self.chunk, self.logits = rows, chunk, logits
        set_polling_wait(polling_wait)
        start = time.perf_counter()
        # off the AMD kernel path upstream re-decodes the packed weights on every call unless they are realized
        if realize is None:
            realize = not default_device().startswith("AMD")
        self.model, self.kv = Transformer.from_gguf(str(model_path), max_context, realize=realize)
        if logits:
            self.model.__class__ = _LogitsTransformer
            self.model.prefill_jit = TinyJit(self.model.forward)
            self.model.rollout_jit = TinyJit(self.model.forward)
        self.load_seconds = time.perf_counter() - start
        self.max_context = int(self.model.max_context)
        self.vocab_size = int(self.model.config.vocab_size)
        self._v_pos = UOp.variable("start_pos", 0, self.max_context - 1)
        self._v_toks = UOp.variable("toks", 1, chunk)
        self._temp = Tensor([0.0])
        self._sampled: Any = None  # device tensor holding the ids sampled by the last call (sample mode)
        self.pos = 0
        self.trace: list[StepRecord] = []

    def tokenizer(self) -> Any:
        return SimpleTokenizer.from_gguf_kv(self.kv)

    def reset(self) -> None:
        """Start over at position 0. Upstream clears recurrent state itself when a call starts there."""
        self.pos, self._sampled = 0, None

    def _submit(self, tokens: Any, n_tokens: int) -> Any:
        """Queue one model call and return its output tensor without waiting for the device."""
        out = self.model(tokens, (self._v_pos.bind(self.pos),), self._temp).realize()
        self.pos += n_tokens
        return out

    def _read(self, out: Any) -> Any:
        """Wait for `out` and bring it to the host."""
        if self.logits:
            result = out.numpy()
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
            lo, count = self._v_pos.bind(self.pos), self._v_toks.bind(n)
            # slicing by bound variables lets every chunk and every position reuse one compiled program
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
        return self._run("decode", Tensor(ids, dtype="int32").reshape(self.rows, 1), 1)
