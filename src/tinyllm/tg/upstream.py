"""Adapter over tinygrad's own LLM engine (`tinygrad.llm`).

This is the phase 0 and phase 1 engine, and the oracle every later engine is checked
against. It runs one batch of rows that all sit at the same position, which is what
upstream supports for Qwen3.8's recurrent layers.

Two output modes:
  sample: upstream's graph unchanged; sampling happens on the device and token ids come back.
  logits: the same graph up to the output head; full-vocabulary logits come back to the host.

In logits mode a session can also decode speculatively with the GGUF's MTP block (`speculative=K`,
`UpstreamSession.speculate`, tinyllm.tg.speculative).
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad.llm.model import GatedDeltaNetBlock, Transformer, TransformerBlock
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
from tinyllm.tg import mtp as tl_mtp
from tinyllm.tg import speculative as spec
from tinyllm.tg.dflash import DFlash
from tinyllm.tg.kernels.topk import topk
from tinyllm.tg.sampler import OUTPUT, PARAMS, sample_row

# "0" sends every prefill chunk through the variable-length program, "1" gives full chunks their own fixed-length
# program; unset, the latter on AMD
STATIC_CHUNKS_ENV = "TINYLLM_STATIC_CHUNKS"
# the head the MTP drafts with: "q4" (a Q4_0 copy, the default), "output" (the model's own) or "q4:<rows>" (a Q4_0
# copy of the first rows only, for English and code); see tinyllm.tg.mtp.draft_head
DRAFT_HEAD_ENV = "TINYLLM_DRAFT_HEAD"


class _LogitsTransformer(Transformer):  # type: ignore[misc]
    """Upstream's Transformer with the in-graph sampler removed.

    `forward` mirrors `Transformer.forward` at the pinned commit up to the logits and must be
    re-checked on every tinygrad bump. tests/test_upstream.py asserts that argmax of these
    logits equals upstream's own greedy token.
    """

    taps: Any = None  # (layer indices, inject) when a DFlash2 drafter reads the inputs of those layers

    def forward(self, tokens: Any, temperature: Any, *start_pos: Any) -> Any:
        x = self.token_embd(t := tokens.to(self.token_embd.weight.device)).float()
        if (hc := self.config.hc) is not None:
            x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
        elif self.config.attn_res_block_size:
            x = x.unsqueeze(2)
        idx_mask, features = None, []
        for i, block in enumerate(self.blk):
            if self.taps is not None and i in self.taps[0]:
                features.append(x)
            x, idx_mask = block(x, start_pos, idx_mask, t)
        if hc is not None:
            x = self.output_hc.mix(x)[0] if hasattr(self, "output_hc") else x.mean(2)
        elif self.config.attn_res_block_size:
            x = self.output_res_score.mix(x)
        logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :].float().to(tokens.device)
        if self.taps is not None:
            Tensor.realize(logits, self.taps[1](features[0].cat(*features[1:], dim=-1), start_pos[0]))
        return logits


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

    _restore_row: int | None = None  # speculative: a verify row whose recurrent state must be put back first
    _spec_prev: tuple[int, int] | None = None  # speculative: (first position, accepted row) of the rows in `_vh`

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
        speculative: int = 0,
        drafter: str | Path | None = None,
        checkpoints: int = 0,
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
        self.mtp: Any = None  # the MTP block, when it drafts
        self.dflash: Any = None  # the DFlash2 drafter, when it drafts
        if speculative:
            if not logits or rows != 1:
                raise ValueError("speculative decoding needs a one-row session in logits mode")
            if drafter is None:
                self.model, self.kv, self.mtp = tl_mtp.load(str(model_path), max_context, realize)
                if self.mtp is None:
                    raise ValueError(f"{model_path} has no MTP block to draft with")
            else:
                self.model, self.kv = Transformer.from_gguf(str(model_path), max_context, realize=realize)
                self.dflash = DFlash.load(drafter, int(self.model.max_context))
                if speculative >= self.dflash.block_size:
                    raise ValueError(f"DFlash2 drafts at most {self.dflash.block_size - 1} tokens per block")
        else:
            self.model, self.kv = Transformer.from_gguf(str(model_path), max_context, realize=realize)
        drafting = self.mtp is not None or self.dflash is not None
        self.speculative = speculative if drafting else 0  # K, the drafts per cycle
        forward = self._prefill_mtp if self.mtp is not None else self._prefill_dflash if self.dflash else None
        if logits:
            self.model.__class__ = _LogitsTransformer
            self.model.prefill_jit = TinyJit(forward or self.model.forward)
            self.model.rollout_jit = TinyJit(self.model.forward)
            self._sample_jit = TinyJit(self._sampled_forward)  # decode with the sampler in the program
        # full prefill chunks through a program compiled for exactly `chunk` tokens: with a symbolic token count
        # tinygrad gives elementwise kernels no workgroup, which makes the FFN gate several times slower
        if static_chunks is None:  # on by default where tinyllm's kernels run; the CPU only serves tests
            setting = os.environ.get(STATIC_CHUNKS_ENV)
            static_chunks = default_device().startswith("AMD") if setting is None else setting not in ("", "0")
        self._static_jit = TinyJit(forward or self.model.forward) if static_chunks and chunk > 1 else None
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
        if self.speculative:
            self._init_speculative()
        # copies of the recurrent state at chosen positions (`save_state`, `load_state`); 0 where the model keeps
        # state that is neither indexed by position nor in DeltaNet blocks
        self.checkpoints = checkpoints if checkpoints > 0 and self._checkpointable() else 0
        self._slots: list[list[Any]] = []  # per checkpoint, one buffer per tensor of `_state_tensors`
        self._save_jit, self._load_jit = TinyJit(self._copy_state), TinyJit(self._paste_state)

    def tokenizer(self) -> Any:
        return SimpleTokenizer.from_gguf_kv(self.kv)

    def reset(self) -> None:
        """Start over at position 0. Upstream clears recurrent state itself when a call starts there."""
        self.pos, self._sampled = 0, None
        if self.speculative:
            self._restore_row, self._spec_prev = None, None

    # ---- recurrent-state checkpoints (tinyllm.runner.checkpoints) ----

    def _checkpointable(self) -> bool:
        """Whether every piece of state that is not indexed by position is a DeltaNet block's (Qwen3.8's layout):
        attention caches need no copy to go back to an earlier position, the recurrent state does."""
        ssm = self.model.config.ssm
        for b in self.model.blk:
            if hasattr(b, "ple"):
                return False
            if isinstance(b, GatedDeltaNetBlock):
                if ssm is None or ssm.kda:
                    return False
            elif type(b) not in (TransformerBlock, spec.SpecAttentionBlock):
                return False
        return True

    def _state_tensors(self) -> list[Any]:
        """The DeltaNet blocks' recurrent and convolution states (none for a model without such blocks)."""
        blocks = [b for b in self.model.blk if isinstance(b, GatedDeltaNetBlock)]
        if blocks and not hasattr(blocks[0], "recurrent_state"):
            raise RuntimeError("the model has not run yet: there is no state to checkpoint")
        return [t for b in blocks for t in (b.recurrent_state, b.conv_state)]

    @property
    def checkpoint_bytes(self) -> int:
        """The size of one checkpoint, 0 before the model has run."""
        try:
            return sum(int(t.nbytes()) for t in self._state_tensors())
        except RuntimeError:
            return 0

    def _copy_state(self, *slot: Any) -> Any:
        writes = [d.assign(s) for d, s in zip(slot, self._state_tensors(), strict=True)]
        Tensor.realize(*writes)
        return writes[0]

    def _paste_state(self, *slot: Any) -> Any:
        writes = [s.assign(d) for s, d in zip(self._state_tensors(), slot, strict=True)]
        Tensor.realize(*writes)
        return writes[0]

    def save_state(self, slot: int) -> None:
        """Queue a copy of the recurrent state, the state after the first `pos` tokens, into checkpoint `slot`.
        The buffers of every slot are allocated on first use (`checkpoint_bytes` each)."""
        if not 0 <= slot < self.checkpoints:
            raise ValueError(f"checkpoint slot {slot} outside the session's {self.checkpoints}")
        self._settle()  # a rollback still pending belongs to the state being copied
        states = self._state_tensors()
        if not states:  # attention only: going back costs nothing
            return
        if not self._slots:
            self._slots = [
                [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device).contiguous().realize() for t in states]
                for _ in range(self.checkpoints)
            ]
        self._save_jit(*self._slots[slot])

    def load_state(self, slot: int, pos: int) -> None:
        """Put back checkpoint `slot`, taken at position `pos`: the next call continues from there. The attention
        caches past `pos` are masked out and overwritten as the new tokens arrive, so they need nothing."""
        if not 0 < pos < self.max_context:
            raise ValueError(f"cannot resume at position {pos}")
        self._restore_row = None  # a rollback still pending belongs to the state being replaced
        if self._state_tensors():
            if not self._slots:
                raise ValueError("no checkpoint has been saved")
            self._load_jit(*self._slots[slot])
        self.pos, self._sampled = pos, None
        if self.speculative:
            self._spec_prev = None  # the next prefill or decode call sets it

    # ---- speculative decoding (tinyllm.tg.speculative) ----

    def _init_speculative(self) -> None:
        cfg, k = self.model.config, self.speculative
        if cfg.hc is not None or cfg.attn_res_block_size:
            raise ValueError("speculative decoding supports plain residual models only")
        setting = os.environ.get(DRAFT_HEAD_ENV, "q4").strip().lower()
        if setting == "output":
            self._draft_head = self.model.output
        elif setting == "q4" or setting.startswith("q4:"):
            rows = int(setting[3:]) if setting.startswith("q4:") else None
            self._draft_head = tl_mtp.draft_head(self.model.output, rows)
        else:
            raise ValueError(f"{DRAFT_HEAD_ENV}: expected output, q4 or q4:<rows>, got {setting!r}")
        self._recurrent = spec.recurrent_blocks(self.model)
        self._verify_blocks = [*self._recurrent, *spec.attention_blocks(self.model)]
        for b in self._recurrent:
            b.allocate(k + 1, default_device())
        if self.dflash is not None:
            # every row the main model processes hands its layer inputs to the drafter (`_LogitsTransformer.taps`)
            self.model.taps = (frozenset(self.dflash.target_layers), self._inject)
            self._spec_io = (_mapped(1 + (k + 1) * PARAMS), _mapped((k + 1) * OUTPUT + 1))
            self._cycle_jit = TinyJit(self._cycle_dflash)
        else:
            # the main model's output-normed hidden at each row of the last verify (after a prefill: its last row)
            self._vh = Tensor.zeros(1, k + 1, cfg.dim).contiguous().realize()
            self._spec_io = (_mapped(k + 2 + (k + 1) * PARAMS), _mapped((k + 1) * OUTPUT + 1))
            self._mtp_tokens = np.zeros(k + 1, dtype=np.float32)  # the tokens the main model chose at those rows
            self._cycle_jit = TinyJit(self._cycle)
        self._restore_jits = [TinyJit(functools.partial(self._restore, n)) for n in range(k)]
        self._v_prev = UOp.variable("spec_prev", 0, self.max_context - 1)
        self._v_row = UOp.variable("spec_row", 0, k)
        self._v_verify = UOp.variable("spec_pos", 1, self.max_context - 1)
        self._restore_row, self._spec_prev = None, None

    def _blocks(self, tokens: Any, start_pos: tuple[Any, ...], features: list[Any] | None = None) -> Any:
        """The main model's layers and output norm over `tokens` (1, T): the hidden of every row. With `features`,
        the inputs of the DFlash2 drafter's target layers are appended to it."""
        x = self.model.token_embd(tokens).float()
        taps = self.dflash.target_layers if features is not None else ()
        for i, block in enumerate(self.model.blk):
            if i in taps:
                features.append(x)  # type: ignore[union-attr]
            x, _ = block(x, start_pos, None, tokens)
        return self.model.output_norm(x)

    def _inject(self, features: Any, start: Any) -> Any:
        """The drafter's KV for rows whose target-layer inputs are `features`, from position `start` on."""
        stores = self.dflash.inject(features, start)
        return Tensor(self.dflash.cache[0].uop.after(*stores))

    def _prefill_dflash(self, tokens: Any, following: Any, temperature: Any, *start_pos: Any) -> Any:
        """A prefill call that also writes the DFlash2 drafter's KV for these rows (`following` is not needed)."""
        features: list[Any] = []
        h = self._blocks(tokens, start_pos, features)
        logits = self.model.output(h[:, -1:])[:, -1, :].float()
        Tensor.realize(logits, self._inject(features[0].cat(*features[1:], dim=-1), start_pos[0]))
        return logits

    def _cycle_dflash(self, io_in: Any, io_out: Any, pos: Any) -> Any:
        """Draft a block with DFlash2 at `pos`, verify it, and inject the verified rows for the next block."""
        k, model, dflash = self.speculative, self.model, self.dflash
        last = io_in[0:1].cast(dtypes.int32).reshape(1, 1)
        params = io_in[1:].reshape(k + 1, PARAMS)
        block = last.cat(Tensor.full((1, k), dflash.mask_token, dtype=dtypes.int32), dim=1)
        hidden = dflash.hidden(model, block, pos)[0, 1:]
        drafts = dflash.select(hidden, self._draft_head(hidden).float(), last.reshape(1))
        drafts.realize()  # the block's KV entries are overwritten by the injection below
        verify = last.cat(drafts, dim=1)
        features: list[Any] = []
        with spec.recording(self._verify_blocks):
            h = self._blocks(verify, (pos,), features)
        logits = model.output(h).float()[0]
        rows = Tensor.stack(*(sample_row(logits[j], params[j]) for j in range(k + 1)))
        n = spec.leading_matches(rows[:, 0], verify[0, 1:].float())
        out = io_out.assign(rows.flatten().cat(n.float().reshape(1)))
        Tensor.realize(out, self._inject(features[0].cat(*features[1:], dim=-1), pos))
        return out

    def _prefill_mtp(self, tokens: Any, following: Any, temperature: Any, *start_pos: Any) -> Any:
        """A prefill call that also writes the MTP block's KV for these rows (each row's hidden paired with the
        token after it, `following`) and keeps the last row's hidden for the first draft."""
        h = self._blocks(tokens, start_pos)
        logits = self.model.output(h[:, -1:])[:, -1, :].float()
        g = self.mtp.hidden(self.model, following, h, start_pos)
        last = h[:, -1:].float().pad((None, (0, self.speculative), None))
        Tensor.realize(logits, g, self._vh.assign(last))
        return logits

    def _restore(self, row: int) -> Any:
        writes = [w for b in self._recurrent for w in b.restore_from(row)]
        Tensor.realize(*writes)
        return writes[0]

    def _cycle(self, io_in: Any, io_out: Any, prev: Any, row: Any, pos: Any) -> Any:
        k, model = self.speculative, self.model
        toks = io_in[: k + 1].cast(dtypes.int32).reshape(1, k + 1)
        last = io_in[k + 1 : k + 2].cast(dtypes.int32).reshape(1, 1)
        params = io_in[k + 2 :].reshape(k + 1, PARAMS)
        # the MTP over the previous verify's rows: KV for the accepted ones, and the first draft from the last
        g = self.mtp.hidden(model, toks, self._vh, (prev,)).shrink((None, (row, row + 1), None))

        def draft(g: Any) -> Any:  # the most likely token, by the top-k kernel (tinygrad's argmax is generic code)
            return topk(self._draft_head(g).reshape(-1).float(), 1)[1].reshape(1, 1)

        drafts = [draft(g)]
        for j in range(1, k):
            g = self.mtp.hidden(model, drafts[-1], g, (pos if j == 1 else pos + (j - 1),))
            drafts.append(draft(g))
        Tensor.realize(*drafts)  # reads `_vh` before the verify overwrites it
        verify = last.cat(*drafts, dim=1)
        with spec.recording(self._verify_blocks):
            h = self._blocks(verify, (pos,))
        logits = model.output(h).float()[0]
        rows = Tensor.stack(*(sample_row(logits[j], params[j]) for j in range(k + 1)))
        n = spec.leading_matches(rows[:, 0], verify[0, 1:].float())
        packed = rows.flatten().cat(n.float().reshape(1))
        out = io_out.assign(packed)
        Tensor.realize(out, self._vh.assign(h.float()))
        return out

    def _settle(self) -> None:
        """Put back the recurrent state a partly rejected verify left, before anything else runs."""
        if self.speculative and self._restore_row is not None:
            self._restore_jits[self._restore_row]()
            self._restore_row = None

    def _after_plain_step(self) -> None:
        """A one-token call without the MTP: drafting can resume after it, from a hidden that is one call old
        (the first draft of the next cycle is then likely rejected; the output is unaffected)."""
        if self.speculative:
            self._spec_prev = (self.pos, 0)

    def speculate(self, token: int, params: Any) -> tuple[Any, int]:
        """One draft-and-verify cycle after `token` (sampled, not yet fed): (K + 1, OUTPUT) sampler results, one
        per verify row, and n, how many drafts were accepted. Rows 0..n hold the n + 1 tokens produced; the state
        then holds `token` and the n accepted drafts. `params`: K + 1 rows of the device sampler's settings.
        The returned array is reused by the next call."""
        k = self.speculative
        if not k:
            raise ValueError("this session was not built for speculative decoding")
        if self.pos + k + 1 >= self.max_context:
            raise ValueError(f"a cycle of {k + 1} tokens does not fit the context of {self.max_context}")
        if self._spec_prev is None:
            raise ValueError("speculate needs a prefill first")
        GlobalCounters.reset()
        start = time.perf_counter()
        self._settle()
        prev, row = self._spec_prev
        (io_in, in_view), (io_out, out_view) = self._spec_io
        settings = np.asarray(params, dtype=np.float32).reshape(-1)
        if self.dflash is not None:
            in_view[0] = float(token)
            in_view[1:] = settings
            self._cycle_jit(io_in, io_out, self._v_verify.bind(self.pos))
        else:
            self._mtp_tokens[row] = token
            in_view[: k + 1] = self._mtp_tokens
            in_view[k + 1] = float(token)
            in_view[k + 2 :] = settings
            b = self._v_prev.bind(prev), self._v_row.bind(row), self._v_verify.bind(self.pos)
            self._cycle_jit(io_in, io_out, *b)
        synchronize()
        n = int(out_view[-1])
        rows = out_view[:-1].reshape(k + 1, OUTPUT)
        if self.dflash is None:
            self._mtp_tokens[:] = rows[:, 0]
        self._spec_prev = (self.pos, n)
        self._restore_row = n if n < k else None
        self.pos += n + 1
        seconds = time.perf_counter() - start
        self.trace.append(StepRecord("speculate", self.rows, n + 1, seconds, int(GlobalCounters.global_mem)))
        return rows, n

    def _submit(self, tokens: Any, n_tokens: int, following: Any = None) -> Any:
        """Queue one model call and return its output tensor without waiting for the device."""
        self._settle()
        full = isinstance(tokens.shape[1], int) and tokens.shape[1] == self.chunk
        if self.speculative:
            if following is None:  # the prefill programs of a speculative session also fill the drafter's KV
                raise ValueError("a speculative session's prefill calls need the following tokens")
            jit = self._static_jit if self._static_jit is not None and full else self.model.prefill_jit
            name = "start_pos_0" if jit is self._static_jit else "start_pos"
            pos = UOp.variable(name, 0, self.max_context - 1).bind(self.pos)
            out = jit(tokens.contiguous(), following.contiguous(), self._temp, pos).realize()
            self._spec_prev = (self.pos + n_tokens - 1, 0)
            self.pos += n_tokens
            return out
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

    def _run(self, kind: str, tokens: Any, n_tokens: int, following: Any = None) -> Any:
        GlobalCounters.reset()
        start = time.perf_counter()
        result = self._read(self._submit(tokens, n_tokens, following))
        self.trace.append(
            StepRecord(kind, self.rows, n_tokens, time.perf_counter() - start, int(GlobalCounters.global_mem))
        )
        return result

    def prefill(
        self,
        tokens: list[int],
        wait_each: bool = True,
        following: int | None = None,
        before_last: Callable[[], None] | None = None,
    ) -> Any:
        """Feed `tokens` to every row, `chunk` tokens per call. Returns the output of the last call (see `step`).

        `before_last` is called before the last call is queued, with `pos` at its first token (where a caller
        takes a checkpoint of the state, `save_state`).

        With speculative decoding the MTP block's KV is written for these rows too: each row pairs with the token
        after it, which for the last row is `following` when the caller knows it (the next slice of a prompt),
        and otherwise the token sampled next, which the first speculative cycle supplies.

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
        nxt = None
        if self.speculative:  # the same table shifted by one: row i holds the token at i + 1
            shifted = np.zeros_like(buf)
            shifted[:, :-1] = buf[:, 1:]
            if following is not None:
                shifted[:, end - 1] = following
            nxt = Tensor(shifted)
        out, calls = None, 0
        GlobalCounters.reset()
        start = time.perf_counter()
        while self.pos < end:
            n = min(self.chunk, end - self.pos)
            if before_last is not None and self.pos + n == end:
                before_last()
            lo = self._v_pos.bind(self.pos)
            # slicing by bound variables lets every chunk and every position reuse one compiled program;
            # with static chunks a full chunk has a fixed length and goes to its own program
            count = n if self._static_jit is not None and n == self.chunk else self._v_toks.bind(n)
            following_rows = None if nxt is None else nxt[:, lo : lo + count]
            if wait_each:
                out = self._run("prefill", table[:, lo : lo + count], n, following_rows)
            else:
                out, calls = self._submit(table[:, lo : lo + count], n, following_rows), calls + 1
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
        self._settle()
        self._after_plain_step()
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
        self._settle()
        self._after_plain_step()
        self._write_decode_input(ids)
        out = self.model.rollout_jit(self._decode_in, self._temp, self._v_pos.bind(self.pos))
        self.pos += 1
        result = self._read(out)
        self.trace.append(
            StepRecord("decode", self.rows, 1, time.perf_counter() - start, int(GlobalCounters.global_mem))
        )
        return result
