"""Adapter over tinygrad's own LLM engine (`tinygrad.llm`).

This is the phase 0 and phase 1 engine, and the oracle every later engine is checked
against. It runs one batch of rows that all sit at the same position, which is what
upstream supports for Qwen3.8's recurrent layers.

Two output modes:
  sample: upstream's graph unchanged; sampling happens on the device and token ids come back.
  logits: the same graph up to the output head; full-vocabulary logits come back to the host.

In logits mode a session can also decode speculatively with the model's MTP block (`speculative=K`,
`UpstreamSession.speculate`, tinyllm.tg.speculative): the GGUF's own (Qwen3.8), or one in a GGUF of its own
(`mtp_path`, Qwen3.8-Flash-Next).
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad.llm.gguf import GGUFPackedTable
from tinygrad.llm.model import GatedDeltaNetBlock, Transformer, TransformerBlock
from tinygrad.llm.tokenizer import SimpleTokenizer

from tinyllm.tg import (
    Buffer,
    BufferSpec,
    GlobalCounters,
    Tensor,
    UOp,
    default_device,
    dtypes,
    set_polling_wait,
    synchronize,
)
from tinyllm.tg import kernels as tl_kernels
from tinyllm.tg import lowering as tl_lowering
from tinyllm.tg import mtp as tl_mtp
from tinyllm.tg import ngram as tl_ngram
from tinyllm.tg import programs as tl_programs
from tinyllm.tg import speculative as spec
from tinyllm.tg.dflash import DFlash
from tinyllm.tg.kernels.moe_small import install_decoders
from tinyllm.tg.kernels.topk import topk
from tinyllm.tg.sampler import OUTPUT, PARAMS, sample_row

install_decoders()  # Q2_0, which tinygrad at the pinned commit cannot read (Qwen3.8-Flash-Next UD-IQ4_XS)

# "0" sends every prefill chunk through the variable-length program, "1" gives full chunks their own fixed-length
# program; unset, the latter on AMD
STATIC_CHUNKS_ENV = "TINYLLM_STATIC_CHUNKS"
SMALL_CHUNK = 256  # the largest call of the short-call program, for sessions with chunks above twice this
# the largest calls of the programs for partial chunks, the short-call program first: a call goes to the smallest
# that takes it, so it pays for at most twice its length instead of for a whole chunk. Longer calls that are not of
# one of STATIC_SIZES run the program for up to a whole chunk; programs for up to 1,024 and 1,536 tokens were
# dropped for their start-up time (about 140 s each at every start)
PARTIAL_SIZES = (SMALL_CHUNK, 512)
# With large chunks, calls of exactly these lengths (a prompt's stretch between two checkpoint positions is a multiple
# of 256) have fixed-length programs of their own, like a full chunk. tinygrad cannot split or tune a generic kernel
# along a symbolic axis, and those kernels run twice as slow per token in the programs above: a 1,536-token call
# takes 1,735 ms here against 1,971. Each program adds to the start-up time. "0" turns them off; a comma-separated
# list names other lengths.
# Where the model drafts with an MTP block these programs also take calls that are padded (`_prefill_padded`): a
# prompt's last stretch of more than 256 tokens runs as one call of the next of these lengths, with the checkpoint
# taken inside it, 256 rows before its end. Before, that stretch was a call up to the checkpoint and a short one
# after it, and the short one read most of the experts again: 0.37 to 0.49 s whatever its length
STATIC_SIZES, STATIC_SIZES_ENV = (512, 768, 1024, 1280, 1536, 1792), "TINYLLM_STATIC_SIZES"
# the head the MTP drafts with: "q4" (a Q4_0 copy, the default), "output" (the model's own) or "q4:<rows>" (a Q4_0
# copy of the first rows only, for English and code); see tinyllm.tg.mtp.draft_head
DRAFT_HEAD_ENV = "TINYLLM_DRAFT_HEAD"
# Copied blocks (`UpstreamSession.speculate`): the tokens before the block that must also precede its source, how far
# back a longer match is looked for to choose between sources, and the cycles without copies after a block of which
# fewer than half the drafts were accepted
COPY_MATCH, COPY_LONG, COPY_PAUSE = 16, 64, 8


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
                features.append(x if hc is None else x.mean(2))  # a drafter for residual streams reads their mean
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


def prefill_call(pos: int, end: int, chunk: int, tail: int, padded: tuple[int, ...], limit: int) -> tuple[int, int]:
    """The prefill call that starts at `pos` of a stretch that ends at `end`: its tokens, and the length it is
    padded to (0: not padded).

    Without padded programs (`padded` empty) calls take `chunk` tokens, cut at `tail`, the position of the
    checkpoint before the last call (0: none). With them the last call, when it has more than 256 tokens, runs
    padded to the next multiple of 256 if a program of that length exists and its rows fit below `limit`; the calls
    before it are whole chunks, and a last call of at most 256 tokens runs as it is."""
    n = min(chunk, end - pos)
    if padded and pos + n == end and n > SMALL_CHUNK:
        length = -(-n // SMALL_CHUNK) * SMALL_CHUNK
        if length in padded and pos + length <= limit:
            return n, length
    elif padded:
        return n, 0
    return (min(n, tail - pos) if pos < tail else n), 0


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
    # speculative: the last call was a copied block's verify (which leaves the MTP block's cache complete), and
    # every one of its drafts was accepted (the next block of the run needs no confirming draft)
    _copy_prev: tuple[bool, bool] = (False, False)
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
        small_chunk: int | None = None,
        mtp_path: str | Path | None = None,
        copy: int = 0,
        programs: str | Path | None = None,
    ) -> None:
        self.rows, self.chunk, self.logits = rows, chunk, logits
        # the compiled programs, kept in the directory `programs` between starts (tinyllm.tg.programs); None
        # builds them at every start. The session itself and the n-gram row reader alias buffers from the host:
        # they create those here, in the constructor, and their attributes are not saved
        self.programs = tl_programs.Programs(
            self,
            Path(programs).expanduser() if programs else None,
            (UpstreamSession, tl_ngram.FileRows, GGUFPackedTable),
        )
        jit = self.programs.jit
        tl_lowering.install()  # a program's tuned kernels are lowered in worker processes
        # drafts of a block copied from earlier text (0: none): its verify runs copy + 1 rows on the matrix cores
        if copy and (copy + 1) % 16:
            raise ValueError(f"a copied block's verify needs a multiple of 16 rows, not {copy + 1}")
        self._copy_request = copy
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
                if self.mtp is None and mtp_path is not None:  # the block ships in a file of its own
                    self.mtp = tl_mtp.load_streams(self.model, str(mtp_path), realize)
                if self.mtp is None:
                    raise ValueError(f"{model_path} has no MTP block to draft with")
            else:
                self.model, self.kv = Transformer.from_gguf(str(model_path), max_context, realize=realize)
                self.dflash = DFlash.load(drafter, int(self.model.max_context))
                if speculative >= self.dflash.block_size:
                    raise ValueError(f"DFlash2 drafts at most {self.dflash.block_size - 1} tokens per block")
                cfg = self.model.config
                if self.dflash.dim != cfg.dim or max(self.dflash.target_layers) >= cfg.num_blocks:
                    raise ValueError(f"{drafter} was not made for {model_path}: width or layers differ")
        else:
            self.model, self.kv = Transformer.from_gguf(str(model_path), max_context, realize=realize)
        drafting = self.mtp is not None or self.dflash is not None
        self.speculative = speculative if drafting else 0  # K, the drafts per cycle
        forward = self._prefill_mtp if self.mtp is not None else self._prefill_dflash if self.dflash else None
        if logits:
            self.model.__class__ = _LogitsTransformer
            self.model.prefill_jit = jit("prefill", forward or self.model.forward)
            self.model.rollout_jit = jit("decode", self.model.forward)
            self._sample_jit = jit("decode sampled", self._sampled_forward)  # decode with the sampler in the program
        # full prefill chunks through a program compiled for exactly `chunk` tokens: with a symbolic token count
        # tinygrad gives elementwise kernels no workgroup, which makes the FFN gate several times slower
        if static_chunks is None:  # on by default where tinyllm's kernels run; the CPU only serves tests
            setting = os.environ.get(STATIC_CHUNKS_ENV)
            static_chunks = default_device().startswith("AMD") if setting is None else setting not in ("", "0")
        static = static_chunks and chunk > 1
        # with an MTP block, fixed-length programs take a count of real rows and may be padded (`_prefill_padded`)
        self.pads = self.mtp is not None and bool(static) and chunk > 2 * SMALL_CHUNK
        self.pad_last = True  # False: plan a prompt's last stretch as two calls all the same (to compare the two)
        fixed = self._prefill_padded if self.pads else forward or self.model.forward
        self._static_jit = jit(f"prefill {chunk}", fixed) if static else None
        self.static_chunks = self._static_jit is not None
        # fixed-length programs for the shorter calls on the checkpoint grid, by length (the full chunk's among them)
        lengths: tuple[int, ...] = ()
        if self._static_jit is not None and chunk > 2 * SMALL_CHUNK:
            setting = os.environ.get(STATIC_SIZES_ENV)
            if setting is None:
                lengths = tuple(n for n in STATIC_SIZES if n < chunk)
            else:
                lengths = tuple(int(n) for n in setting.split(",") if n.strip() and 0 < int(n) < chunk)
        self.static_sizes = lengths
        # the same function at other lengths: once the full chunk's program is captured, each of these is captured
        # by its first call (tinyllm.tg.programs.Program), as are the later partial and restore programs below
        self._static: dict[int, Any] = {n: jit(f"prefill {n}", fixed, self._static_jit) for n in lengths}
        if self._static_jit is not None:
            self._static[chunk] = self._static_jit
        # with large chunks, calls shorter than a chunk (the rest of a prompt) go to programs compiled for at most
        # 256 and 512 tokens: tinyllm's kernels compute a symbolic chunk at its largest length, so 454 tokens
        # through the program for up to 2,048 cost most of a full call (docs/04-qwen3.8-flash-next.md, section
        # 15). `small_chunk` names the only such program instead (0: none)
        if small_chunk is None:
            sizes = tuple(n for n in PARTIAL_SIZES if n < chunk) if chunk > 2 * SMALL_CHUNK else ()
        else:
            sizes = (small_chunk,) if 0 < small_chunk < chunk else ()
        self.partial_sizes = sizes
        # the smallest: also the grid of the checkpoints taken before a prompt's last call (`prefill`)
        self.small_chunk = sizes[0] if sizes else 0
        self._partial: list[tuple[int, Any, Any]] = []
        for i, n in enumerate(sizes):
            program = jit(f"prefill up to {n}", forward or self.model.forward, self._partial[0][1] if i else None)
            self._partial.append((n, program, UOp.variable("toks_small" if i == 0 else f"toks_{n}", 1, n)))
        if logits and self._partial:  # the program for up to a whole chunk: that function too
            self.model.prefill_jit.like = self._partial[0][1]
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
        # Qwen3.8-Flash-Next's n-gram embedding (tinygrad's PLELayer): each position's rows are copied out of the
        # table on the host before the call that reads them, from the ids fed up to there (`_feed_ple`)
        ple = self.model.config.ple
        self._ple: Any = self.model.blk[ple.layer].ple if ple is not None else None
        if self._ple is not None:
            # rows read by tinyllm, not through tinygrad's huge-page mapping, into a buffer made here
            tl_ngram.install(self._ple, self._ple.max_context + self.chunk)
        self._history: list[int] = []  # with an n-gram embedding: the ids at positions 0..pos
        self._prefetcher: Any = None  # a thread reading the next prefill call's n-gram rows (`_prefetch_ple`)
        self._ple_ahead: Any = None  # (first position, ids up to its end, job) of the rows written ahead
        self._sampled_ids: Any = None  # sample mode: the ids the last call sampled, on the host
        # one past the last position the padding rows of the last `prefill` wrote cache entries to (0: none)
        self.padded_end = 0
        if self._ple is not None and rows != 1:
            raise ValueError("a model with an n-gram embedding runs one row")
        if self.speculative:
            self._init_speculative()
        # copies of the recurrent state at chosen positions (`save_state`, `load_state`); 0 where the model keeps
        # state that is neither indexed by position nor in DeltaNet blocks
        self.checkpoints = checkpoints if checkpoints > 0 and self._checkpointable() else 0
        self._slots: list[list[Any]] = []  # per checkpoint, one buffer per tensor of `_state_tensors`
        self._save_jit, self._load_jit = jit("state out", self._copy_state), jit("state in", self._paste_state)
        self._save_mid_jit = jit("state out of a call", self._copy_mid)
        if self.pads:
            for b in self._rollback:
                b.allocate_mid(default_device())
        files = [*tl_programs.model_files(Path(model_path).expanduser())]
        files += [Path(extra).expanduser() for extra in (mtp_path, drafter) if extra is not None]
        settings = {
            "max_context": max_context,
            "rows": rows,
            "chunk": chunk,
            "logits": logits,
            "realize": realize,
            "polling_wait": polling_wait,
            "kernels": list(self.kernels),
            "static": sorted(self._static),
            "partial": list(self.partial_sizes),
            "speculative": self.speculative,
            "copy": self.copy if self.speculative else 0,
            "mtp": self.mtp is not None,
            "dflash": self.dflash is not None,
            "checkpoints": self.checkpoints,
        }
        self.programs.identify(Path(model_path).name.removesuffix(".gguf"), files, settings)
        self.programs.restore()

    def tokenizer(self) -> Any:
        return SimpleTokenizer.from_gguf_kv(self.kv)

    def reset(self) -> None:
        """Start over at position 0. Upstream clears recurrent state itself when a call starts there."""
        self.pos, self._sampled = 0, None
        self._history.clear()
        if self.speculative:
            self._restore_row, self._spec_prev, self._copy_pause, self._copy_prev = None, None, 0, (False, False)

    # ---- recurrent-state checkpoints (tinyllm.runner.checkpoints) ----

    def _checkpointable(self) -> bool:
        """Whether every piece of state that is not indexed by position is a DeltaNet block's or the n-gram
        embedding's convolution state (Qwen3.8's and Flash-Next's layouts): attention caches, the sparse
        attention's indexer keys and the staged n-gram rows need no copy to go back to an earlier position, the
        recurrent state does."""
        ssm = self.model.config.ssm
        for b in self.model.blk:
            if hasattr(b, "ple") and b.ple is not self._ple:
                return False
            if isinstance(b, GatedDeltaNetBlock):
                if ssm is None or ssm.kda:
                    return False
            elif type(b) not in (TransformerBlock, spec.SpecAttentionBlock):
                return False
        return True

    def _state_tensors(self) -> list[Any]:
        """The DeltaNet blocks' recurrent and convolution states and the n-gram embedding's convolution state
        (none for a model without either)."""
        blocks = [b for b in self.model.blk if isinstance(b, GatedDeltaNetBlock)]
        if (blocks and not hasattr(blocks[0], "recurrent_state")) or (
            self._ple is not None and not hasattr(self._ple, "conv_state")
        ):
            raise RuntimeError("the model has not run yet: there is no state to checkpoint")
        states = [t for b in blocks for t in (b.recurrent_state, b.conv_state)]
        return states + ([self._ple.conv_state] if self._ple is not None else [])

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

    def _mid_tensors(self) -> list[Any]:
        """Where a padded prefill call leaves the state at its checkpoint row, in `_state_tensors`' order."""
        blocks = [b for b in self.model.blk if isinstance(b, GatedDeltaNetBlock)]
        states = [t for b in blocks for t in (b.mid_state, b.mid_conv)]
        return states + ([self._ple.mid_conv] if self._ple is not None else [])

    def _copy_mid(self, *slot: Any) -> Any:
        writes = [d.assign(s) for d, s in zip(slot, self._mid_tensors(), strict=True)]
        Tensor.realize(*writes)
        return writes[0]

    def _slot(self, slot: int) -> list[Any]:
        """Checkpoint `slot`'s buffers; those of every slot are allocated on first use (`checkpoint_bytes` each)."""
        if not 0 <= slot < self.checkpoints:
            raise ValueError(f"checkpoint slot {slot} outside the session's {self.checkpoints}")
        if not self._slots:
            states = self._state_tensors()
            self._slots = [
                [Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device).contiguous().realize() for t in states]
                for _ in range(self.checkpoints)
            ]
        return self._slots[slot]

    def save_mid_state(self, slot: int) -> None:
        """Queue a copy, into checkpoint `slot`, of the state the last padded prefill call kept: the state at the
        start of its last 256 rows (`prefill` names the position)."""
        self._save_mid_jit(*self._slot(slot))

    def save_state(self, slot: int) -> None:
        """Queue a copy of the recurrent state, the state after the first `pos` tokens, into checkpoint `slot`.
        The buffers of every slot are allocated on first use (`checkpoint_bytes` each)."""
        if not 0 <= slot < self.checkpoints:
            raise ValueError(f"checkpoint slot {slot} outside the session's {self.checkpoints}")
        self._settle()  # a rollback still pending belongs to the state being copied
        if not self._state_tensors():  # attention only: going back costs nothing
            return
        self._save_jit(*self._slot(slot))

    def load_state(self, slot: int, pos: int, history: Any = None) -> None:
        """Put back checkpoint `slot`, taken at position `pos`: the next call continues from there. The attention
        caches past `pos` are masked out and overwritten as the new tokens arrive, so they need nothing.
        `history`: the `pos` tokens the checkpoint stands for, when the session may have held others since (the
        runner resumes a conversation after another request)."""
        if not 0 < pos < self.max_context:
            raise ValueError(f"cannot resume at position {pos}")
        self._restore_row = None  # a rollback still pending belongs to the state being replaced
        if self._state_tensors():
            if not self._slots:
                raise ValueError("no checkpoint has been saved")
            self._load_jit(*self._slots[slot])
        # the n-gram rows of the next positions hash the ids before them: the checkpoint's
        if history is not None:
            self._history = [int(t) for t in history[:pos]]
            if self.speculative and self.copy:
                self._text[:pos] = self._history
        else:  # those the session has held since, when the checkpoint lies on them
            del self._history[pos:]
        self.pos, self._sampled = pos, None
        if self.speculative:
            self._spec_prev, self._copy_prev = None, (False, False)  # the next prefill or decode call sets it

    # ---- speculative decoding (tinyllm.tg.speculative) ----

    def _init_speculative(self) -> None:
        cfg, k, jit = self.model.config, self.speculative, self.programs.jit
        streams = cfg.hc is not None  # Qwen3.8-Flash-Next: residual streams, which its MTP block reads and writes
        if cfg.attn_res_block_size or (streams and not cfg.hc.low_rank):
            raise ValueError("speculative decoding supports plain residual models and Qwen3.8-Flash-Next's streams")
        setting = os.environ.get(DRAFT_HEAD_ENV, "q4").strip().lower()
        if setting == "output":
            self._draft_head = self.model.output
        elif setting == "q4" or setting.startswith("q4:"):
            rows = int(setting[3:]) if setting.startswith("q4:") else None
            self._draft_head = tl_mtp.draft_head(self.model.output, rows)
        else:
            raise ValueError(f"{DRAFT_HEAD_ENV}: expected output, q4 or q4:<rows>, got {setting!r}")
        self._recurrent = spec.recurrent_blocks(self.model)
        # what a partly rejected verify puts back: the DeltaNet blocks' states and the n-gram layer's convolution
        self._rollback: list[Any] = [*self._recurrent]
        if self._ple is not None:
            self._rollback.append(spec.ple_layer(self._ple))
        # sparse attention layers (Flash-Next) attend each verify row with decoding's kernel as they are
        full_attention = [] if cfg.indexer is not None else spec.attention_blocks(self.model)
        self._verify_blocks = [*self._rollback, *full_attention]
        # blocks copied from earlier text: with the MTP block and the two-program cycle (the host step between them)
        self.copy = self._copy_request if self.mtp is not None and self._ple is not None else 0
        most = max(k, self.copy)  # the most drafts a verify takes
        self.speculative_rows = most + 1  # rows of sampler settings `speculate` wants
        self.copied = [0, 0, 0]  # copied blocks, their drafts, the drafts accepted
        self.last_drafted = k  # drafts of the last cycle: K, or a copied block's
        self._copy_pause = 0  # cycles left without copied blocks
        for b in self._rollback:
            b.allocate(most + 1, default_device())
        if self.dflash is not None:
            # every row the main model processes hands its layer inputs to the drafter (`_LogitsTransformer.taps`)
            self.model.taps = (frozenset(self.dflash.target_layers), self._inject)
            self._spec_io = (_mapped(1 + (k + 1) * PARAMS), _mapped((k + 1) * OUTPUT + 1))
            self._cycle_jit = jit("cycle", self._cycle_dflash)
            self._draft_jit, self._verify_jit = jit("draft", self._draft_dflash), jit("verify", self._verify_dflash)
        else:
            # what the MTP block reads of the main model at each row of the last verify (after a prefill: its last
            # row): the output-normed hidden, or with residual streams the streams before the final mixer
            shape = (1, k + 1, cfg.hc.count, cfg.dim) if streams else (1, k + 1, cfg.dim)
            self._vh = Tensor.zeros(*shape).contiguous().realize()
            self._spec_io = (_mapped(k + 2 + (k + 1) * PARAMS), _mapped((k + 1) * OUTPUT + 1))
            self._mtp_tokens = np.zeros(k + 1, dtype=np.float32)  # the tokens the main model chose at those rows
            self._cycle_jit = jit("cycle", self._cycle)
            self._draft_jit, self._verify_jit = jit("draft", self._draft), jit("verify", self._verify)
        # with an n-gram embedding the drafts come back to the host, which copies their rows of the table before
        # the verify reads them: two programs a cycle (`_draft_jit`, `_verify_jit`) instead of one (`_cycle_jit`)
        self._drafts_io = _mapped(k)
        self._restore_jits: list[Any] = []
        for n in range(most):
            first = self._restore_jits[0] if n else None
            self._restore_jits.append(jit(f"restore {n}", functools.partial(self._restore, n), first))
        if self.copy:
            c = self.copy
            self._copy_io = (_mapped(1 + c + (c + 1) * PARAMS), _mapped((c + 1) * OUTPUT + 1))
            self._copy_jit = jit("copied block", self._verify_copy)
            self._text = np.zeros(self.max_context + c + 1, dtype=np.int32)  # the ids at positions 0..pos - 1
        self._v_prev = UOp.variable("spec_prev", 0, self.max_context - 1)
        self._v_row = UOp.variable("spec_row", 0, k)
        self._v_verify = UOp.variable("spec_pos", 1, self.max_context - 1)
        self._restore_row, self._spec_prev = None, None

    def _blocks(self, tokens: Any, start_pos: tuple[Any, ...], features: list[Any] | None = None) -> Any:
        """The main model's layers over `tokens` (1, T): the output-normed hidden of every row, or with residual
        streams the streams (1, T, count, D) before the final mixer (`_head` mixes them). With `features`, the
        inputs of the DFlash2 drafter's target layers are appended to it (of streams, their mean)."""
        x = self.model.token_embd(tokens).float()
        if (hc := self.model.config.hc) is not None:
            x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
        taps = self.dflash.target_layers if features is not None else ()
        idx_mask = None
        for i, block in enumerate(self.model.blk):
            if i in taps:
                features.append(x if hc is None else x.mean(2))  # type: ignore[union-attr]
            x, idx_mask = block(x, start_pos, idx_mask, tokens)
        return x if hc is not None else self.model.output_norm(x)

    def _head(self, h: Any) -> Any:
        """What the output head reads of `_blocks`' result."""
        return self.model.output_hc.mix(h)[0] if self.model.config.hc is not None else h

    def _inject(self, features: Any, start: Any) -> Any:
        """The drafter's KV for rows whose target-layer inputs are `features`, from position `start` on."""
        stores = self.dflash.inject(features, start)
        return Tensor(self.dflash.cache[0].uop.after(*stores))

    def _prefill_dflash(self, tokens: Any, following: Any, temperature: Any, *start_pos: Any) -> Any:
        """A prefill call that also writes the DFlash2 drafter's KV for these rows (`following` is not needed)."""
        features: list[Any] = []
        h = self._blocks(tokens, start_pos, features)
        logits = self.model.output(self._head(h[:, -1:]))[:, -1, :].float()
        Tensor.realize(logits, self._inject(features[0].cat(*features[1:], dim=-1), start_pos[0]))
        return logits

    def _block_drafts(self, last: Any, pos: Any) -> Any:
        """DFlash2's K drafts (1, K) int32 for the block that starts with `last` (1, 1) at `pos`."""
        k, dflash = self.speculative, self.dflash
        block = last.cat(Tensor.full((1, k), dflash.mask_token, dtype=dtypes.int32), dim=1)
        hidden = dflash.hidden(self.model, block, pos)[0, 1:]
        return dflash.select(hidden, self._draft_head(hidden).float(), last.reshape(1))

    def _checked_dflash(self, verify: Any, params: Any, io_out: Any, pos: Any) -> Any:
        """The main model over `verify` (1, K + 1), each row sampled and the leading matches counted, and the
        drafter's KV written for those rows (which overwrites the block's own entries)."""
        k = self.speculative
        features: list[Any] = []
        with spec.recording(self._verify_blocks):
            h = self._blocks(verify, (pos,), features)
        logits = self.model.output(self._head(h)).float()[0]
        rows = Tensor.stack(*(sample_row(logits[j], params[j]) for j in range(k + 1)))
        n = spec.leading_matches(rows[:, 0], verify[0, 1:].float())
        out = io_out.assign(rows.flatten().cat(n.float().reshape(1)))
        Tensor.realize(out, self._inject(features[0].cat(*features[1:], dim=-1), pos))
        return out

    def _cycle_dflash(self, io_in: Any, io_out: Any, pos: Any) -> Any:
        """Draft a block with DFlash2 at `pos`, verify it, and inject the verified rows for the next block."""
        k = self.speculative
        last = io_in[0:1].cast(dtypes.int32).reshape(1, 1)
        drafts = self._block_drafts(last, pos)
        drafts.realize()  # the block's KV entries are overwritten by the injection below
        return self._checked_dflash(last.cat(drafts, dim=1), io_in[1:].reshape(k + 1, PARAMS), io_out, pos)

    def _draft_dflash(self, io_in: Any, io_drafts: Any, pos: Any) -> Any:
        """The first program of a cycle whose drafts the host needs before the verify: the drafts as floats."""
        drafts = self._block_drafts(io_in[0:1].cast(dtypes.int32).reshape(1, 1), pos)
        return io_drafts.assign(drafts.reshape(self.speculative).float()).realize()

    def _verify_dflash(self, io_in: Any, io_drafts: Any, io_out: Any, pos: Any) -> Any:
        """The second: `_checked_dflash` over the last token and those drafts."""
        k = self.speculative
        last = io_in[0:1].cast(dtypes.int32).reshape(1, 1)
        verify = last.cat(io_drafts.cast(dtypes.int32).reshape(1, k), dim=1)
        return self._checked_dflash(verify, io_in[1:].reshape(k + 1, PARAMS), io_out, pos)

    def _prefill_mtp(self, tokens: Any, following: Any, temperature: Any, *start_pos: Any) -> Any:
        """A prefill call that also writes the MTP block's KV for these rows (each row's hidden paired with the
        token after it, `following`) and keeps the last row's hidden for the first draft."""
        h = self._blocks(tokens, start_pos)
        logits = self.model.output(self._head(h[:, -1:]))[:, -1, :].float()
        g = self.mtp.hidden(self.model, following, h, start_pos)
        last = h[:, -1:].float().pad(tuple((0, self.speculative) if i == 1 else None for i in range(h.ndim)))
        Tensor.realize(logits, g, self._vh.assign(last))
        return logits

    def _prefill_padded(self, tokens: Any, following: Any, temperature: Any, real: Any, *start_pos: Any) -> Any:
        """`_prefill_mtp` for a call whose first `real` rows (a bound variable) are the prompt's and the others
        padding: the logits and the hidden kept for the first draft are the last real row's, and the DeltaNet
        blocks and the n-gram layer end in their state after that row (`spec.padded`). They also keep the state
        at the start of the call's last 256 rows, where `prefill` has the caller take a checkpoint; with
        `real` at the call's length this is a plain fixed-length call."""
        mid = max(int(tokens.shape[1]) - SMALL_CHUNK, 0)
        with spec.padded(self._rollback, real, mid):
            h = self._blocks(tokens, start_pos)
        last = h[:, real - 1 : real]
        logits = self.model.output(self._head(last))[:, -1, :].float()
        g = self.mtp.hidden(self.model, following, h, start_pos)
        kept = last.float().pad(tuple((0, self.speculative) if i == 1 else None for i in range(h.ndim)))
        Tensor.realize(logits, g, self._vh.assign(kept))
        return logits

    def _restore(self, row: int) -> Any:
        writes = [w for b in self._rollback for w in b.restore_from(row)]
        Tensor.realize(*writes)
        return writes[0]

    def _chain(self, toks: Any, prev: Any, row: Any, pos: Any) -> list[Any]:
        """The K drafts, each (1, 1) int32: the MTP over the previous verify's rows (KV for the accepted ones, the
        first draft from the last of them, `row`), then K - 1 steps on its own output."""
        k, model = self.speculative, self.model
        g = self.mtp.hidden(model, toks, self._vh, (prev,))
        g = g.shrink(tuple((row, row + 1) if i == 1 else None for i in range(g.ndim)))

        def draft(g: Any) -> Any:  # the most likely token, by the top-k kernel (tinygrad's argmax is generic code)
            return topk(self._draft_head(self.mtp.head(g)).reshape(-1).float(), 1)[1].reshape(1, 1)

        drafts = [draft(g)]
        for j in range(1, k):
            g = self.mtp.hidden(model, drafts[-1], g, (pos if j == 1 else pos + (j - 1),))
            drafts.append(draft(g))
        return drafts

    def _checked(self, verify: Any, params: Any, io_out: Any, pos: Any) -> Any:
        """The main model over `verify` (1, K + 1), the last token and the drafts: each row sampled, the count of
        leading drafts that match, and the rows' hiddens kept for the next cycle's MTP."""
        k = self.speculative
        with spec.recording(self._verify_blocks):
            h = self._blocks(verify, (pos,))
        logits = self.model.output(self._head(h)).float()[0]
        rows = Tensor.stack(*(sample_row(logits[j], params[j]) for j in range(k + 1)))
        n = spec.leading_matches(rows[:, 0], verify[0, 1:].float())
        out = io_out.assign(rows.flatten().cat(n.float().reshape(1)))
        Tensor.realize(out, self._vh.assign(h.float()))
        return out

    def _cycle(self, io_in: Any, io_out: Any, prev: Any, row: Any, pos: Any) -> Any:
        k = self.speculative
        toks = io_in[: k + 1].cast(dtypes.int32).reshape(1, k + 1)
        last = io_in[k + 1 : k + 2].cast(dtypes.int32).reshape(1, 1)
        drafts = self._chain(toks, prev, row, pos)
        Tensor.realize(*drafts)  # reads `_vh` before the verify overwrites it
        return self._checked(last.cat(*drafts, dim=1), io_in[k + 2 :].reshape(k + 1, PARAMS), io_out, pos)

    def _draft(self, io_in: Any, io_drafts: Any, prev: Any, row: Any, pos: Any) -> Any:
        """The first program of a cycle whose drafts the host needs before the verify: the drafts as floats."""
        k = self.speculative
        drafts = self._chain(io_in[: k + 1].cast(dtypes.int32).reshape(1, k + 1), prev, row, pos)
        return io_drafts.assign(drafts[0].cat(*drafts[1:], dim=1).reshape(k).float()).realize()

    def _verify(self, io_in: Any, io_drafts: Any, io_out: Any, pos: Any) -> Any:
        """The second: `_checked` over the last token (in `io_in` after the MTP's tokens) and those drafts."""
        k = self.speculative
        last = io_in[k + 1 : k + 2].cast(dtypes.int32).reshape(1, 1)
        verify = last.cat(io_drafts.cast(dtypes.int32).reshape(1, k), dim=1)
        return self._checked(verify, io_in[k + 2 :].reshape(k + 1, PARAMS), io_out, pos)

    def _verify_copy(self, io_in: Any, io_out: Any, pos: Any) -> Any:
        """The verify of a copied block: the last token and `copy` drafts, all in `io_in` before the settings. As
        `_checked`, and then what the next cycle needs, which a draft program would not find: the MTP block's cache
        for these rows (each row's streams with the token the main model chose there), and in `_vh` the streams of
        the last accepted row, as after a prefill."""
        c, k = self.copy, self.speculative
        verify = io_in[: c + 1].cast(dtypes.int32).reshape(1, c + 1)
        params = io_in[c + 1 :].reshape(c + 1, PARAMS)
        with spec.recording(self._verify_blocks):
            h = self._blocks(verify, (pos,))
        logits = self.model.output(self._head(h)).float()[0]
        rows = Tensor.stack(*(sample_row(logits[j], params[j]) for j in range(c + 1)))
        n = spec.leading_matches(rows[:, 0], verify[0, 1:].float())
        g = self.mtp.hidden(self.model, rows[:, 0].cast(dtypes.int32).reshape(1, c + 1), h, (pos,))
        accepted = (Tensor.arange(c + 1) == n).reshape(1, c + 1, *(1 for _ in h.shape[2:]))
        last = accepted.where(h.float(), 0).sum(1, keepdim=True)
        last = last.pad(tuple((0, k) if i == 1 else None for i in range(h.ndim)))
        out = io_out.assign(rows.flatten().cat(n.float().reshape(1)))
        Tensor.realize(out, g, self._vh.assign(last))
        return out

    def _copy_source(self, token: int) -> Any:
        """The `copy` tokens that followed the latest earlier occurrence of the text's last COPY_MATCH tokens (the
        last of them `token`, sampled and not yet fed), or None. Among several occurrences the one that matches
        furthest back wins (up to COPY_LONG tokens), then the latest."""
        c, n, text = self.copy, self.pos, self._text
        last = n - c  # an occurrence ends before this: `c` tokens follow it inside the text
        if last <= COPY_MATCH:
            return None
        ends = np.flatnonzero(text[COPY_MATCH - 1 : last] == token) + (COPY_MATCH - 1)
        for back in range(1, COPY_MATCH):
            if not ends.size:
                return None
            ends = ends[text[ends - back] == text[n - back]]
        back = COPY_MATCH
        while ends.size > 1 and back < min(COPY_LONG, n):
            further = ends[(ends >= back) & (text[np.maximum(ends - back, 0)] == text[n - back])]
            if not further.size:
                break
            ends, back = further, back + 1
        if not ends.size:
            return None
        end = int(ends[-1])
        return text[end + 1 : end + 1 + c]

    def _settle(self) -> None:
        """Put back the recurrent state a partly rejected verify left, before anything else runs."""
        if self.speculative and self._restore_row is not None:
            self._restore_jits[self._restore_row]()
            self._restore_row = None

    def _after_plain_step(self) -> None:
        """A one-token call without the MTP: drafting can resume after it, from a hidden that is one call old
        (the first draft of the next cycle is then likely rejected; the output is unaffected)."""
        if self.speculative:
            self._spec_prev, self._copy_prev = (self.pos, 0), (False, False)

    def speculate(self, token: int, params: Any, copy: Any = None) -> tuple[Any, int]:
        """One draft-and-verify cycle after `token` (sampled, not yet fed): one row of sampler results per verify
        row, and n, how many drafts were accepted. Rows 0..n hold the n + 1 tokens produced; the state then holds
        `token` and the n accepted drafts. `params`: `speculative_rows` rows of the device sampler's settings (a
        cycle uses the first of them). The returned array is reused by the next call.

        A cycle verifies the K drafts of the MTP block or of the drafter. With `copy` set (a session built with
        it), it verifies a block of that many tokens copied from earlier text instead, when the text's last tokens
        occurred before and the MTP block's own drafts are the tokens that followed there: the block then reads the
        dense weights once for 16 rows. After a copied block whose drafts were all accepted, the next block of the
        run is verified without that confirming draft. `last_drafted` says which kind the cycle was. `copy` (a list
        of that many tokens) forces a copied block, for warming up and tests."""
        k = self.speculative
        if not k:
            raise ValueError("this session was not built for speculative decoding")
        if self.pos + self.speculative_rows >= self.max_context:
            raise ValueError(f"a cycle of {self.speculative_rows} tokens does not fit the context")
        if self._spec_prev is None:
            raise ValueError("speculate needs a prefill first")
        GlobalCounters.reset()
        start = time.perf_counter()
        self._settle()
        prev, row = self._spec_prev
        (io_in, in_view), (io_out, out_view) = self._spec_io
        settings = np.asarray(params, dtype=np.float32).reshape(-1)
        if settings.size < self.speculative_rows * PARAMS:
            raise ValueError(f"speculate needs {self.speculative_rows} rows of sampler settings")
        drafts_io, drafts = self._drafts_io
        block = None if copy is None else np.asarray(copy, dtype=np.int32)
        if block is None and self.copy and not self._copy_pause:
            block = self._copy_source(token)
        self._copy_pause = max(0, self._copy_pause - 1)
        most = k
        if self.dflash is not None:
            in_view[0] = float(token)
            in_view[1:] = settings[: (k + 1) * PARAMS]
            at = self._v_verify.bind(self.pos)
            if self._ple is None:
                self._cycle_jit(io_in, io_out, at)
            else:  # the n-gram rows of the verify's positions hash the drafts: the host copies them in between
                self._draft_jit(io_in, drafts_io, at)
                synchronize()
                self._feed_ple([token, *(int(d) for d in drafts)])
                self._verify_jit(io_in, drafts_io, io_out, at)
        else:
            self._mtp_tokens[row] = token
            in_view[: k + 1] = self._mtp_tokens
            in_view[k + 1] = float(token)
            in_view[k + 2 :] = settings[: (k + 1) * PARAMS]
            b = self._v_prev.bind(prev), self._v_row.bind(row), self._v_verify.bind(self.pos)
            was_copy, whole = self._copy_prev
            if self._ple is None:
                self._cycle_jit(io_in, io_out, *b)
            else:
                # after a copied block the MTP block's cache is complete without the draft program; a block is
                # then taken unconfirmed when the one before it was accepted whole (or when the caller forces it)
                if block is None or not was_copy or not (whole or copy is not None):
                    self._draft_jit(io_in, drafts_io, *b)
                    synchronize()
                    # a copied block only when the MTP block, from the model's own state, drafts its first tokens
                    if block is not None and copy is None and not np.array_equal(drafts, block[:k]):
                        block = None
                if block is None:
                    self._feed_ple([token, *(int(d) for d in drafts)])
                    self._verify_jit(io_in, drafts_io, io_out, b[2])
                else:
                    most = self.copy
                    (io_in, in_view), (io_out, out_view) = self._copy_io
                    in_view[0] = float(token)
                    in_view[1 : most + 1] = block
                    in_view[most + 1 :] = settings[: (most + 1) * PARAMS]
                    self._feed_ple([token, *(int(t) for t in block)])
                    self._copy_jit(io_in, io_out, b[2])
        synchronize()
        n = int(out_view[-1])
        rows = out_view[:-1].reshape(most + 1, OUTPUT)
        if most != k:  # a copied block: `_vh` holds its last accepted row alone, as after a prefill
            self._spec_prev, self._copy_prev = (self.pos + n, 0), (True, n == most)
            self.copied = [self.copied[0] + 1, self.copied[1] + most, self.copied[2] + n]
            if copy is None and n < most // 2:
                self._copy_pause = COPY_PAUSE
        else:
            if self.dflash is None:
                self._mtp_tokens[:] = rows[:, 0]
            self._spec_prev, self._copy_prev = (self.pos, n), (False, False)
        self.last_drafted = most
        self._restore_row = n if n < most else None
        self.pos += n + 1
        del self._history[self.pos :]  # with an n-gram embedding: the rejected drafts
        seconds = time.perf_counter() - start
        self.trace.append(StepRecord("speculate", self.rows, n + 1, seconds, int(GlobalCounters.global_mem)))
        return rows, n

    def _prefetch_ple(self, tokens: Any, first: int, start: int, n: int) -> None:
        """In the background, write the n-gram rows of positions start .. start + n - 1 (`tokens` holds the prompt
        from position `first`) into the host table, while the device runs the call before them. The table has a
        slot per position, so the running call's rows are not touched; `_feed_ple` waits for the job."""
        if self._ple is None or n <= 0 or not tl_ngram.PREFETCH or not isinstance(self._ple.table, tl_ngram.FileRows):
            return
        history = self._history + [int(t) for t in tokens[start - first : start - first + n]]  # a copy
        if len(history) != start + n:
            return
        table, ple, slots = self._ple.table, self._ple.ple, self._ple.max_context + self.chunk
        if self._prefetcher is None:
            self._prefetcher = ThreadPoolExecutor(1, thread_name_prefix="ngram-rows-ahead")
        job = self._prefetcher.submit(lambda: table.write(start, tl_ngram.row_ids(ple, history, start, n), slots))
        self._ple_ahead = (start, history, job)

    def _feed_ple(self, ids: Any) -> None:
        """Copy the n-gram embedding's rows for `ids`, the tokens at positions pos.., into its host table, which
        the next call reads (as `Transformer.generate` does before each call), unless `_prefetch_ple` wrote them
        for these same ids. No-op for other models."""
        if self._ple is None:
            return
        if len(self._history) < self.pos:
            raise RuntimeError(f"n-gram history holds {len(self._history)} ids, the session is at {self.pos}")
        del self._history[self.pos :]
        self._history.extend(int(t) for t in ids)
        if self.speculative and self.copy:
            self._text[self.pos : self.pos + len(ids)] = ids
        ahead, self._ple_ahead = self._ple_ahead, None
        if ahead is not None:
            start, history, job = ahead
            job.result()  # the job's rows are in the table (or its error is raised here)
            if start == self.pos and history == self._history:
                return
        if isinstance(self._ple.table, tl_ngram.FileRows):  # numpy hashing (`tl_ngram.row_ids`)
            rows = tl_ngram.row_ids(self._ple.ple, self._history, self.pos, len(ids))
            self._ple.table.write(self.pos, rows, self._ple.max_context + self.chunk)
        else:
            self._ple.write_rows(self._history, self.pos, len(ids), self.chunk)

    def _partial_for(self, n_tokens: int) -> tuple[Any, Any] | None:
        """The program and the token-count variable of a call of `n_tokens` short of a chunk: the smallest partial
        program that takes it, None for a call longer than them all."""
        return next(((jit, var) for size, jit, var in self._partial if n_tokens <= size), None)

    def _submit(self, tokens: Any, n_tokens: int, following: Any = None) -> Any:
        """Queue one model call and return its output tensor without waiting for the device."""
        self._settle()
        # a fixed-length call (a full chunk, or one of `static_sizes`) runs its own program
        fixed = self._static.get(tokens.shape[1]) if isinstance(tokens.shape[1], int) else None
        full = fixed is not None
        partial = None if full else self._partial_for(n_tokens)
        if self.speculative:
            if following is None:  # the prefill programs of a speculative session also fill the drafter's KV
                raise ValueError("a speculative session's prefill calls need the following tokens")
            jit = fixed if fixed is not None else self.model.prefill_jit
            if partial is not None:
                jit = partial[0]
            assert jit is not None
            name = "start_pos" if jit is self.model.prefill_jit else "start_pos_0"
            pos = UOp.variable(name, 0, self.max_context - 1).bind(self.pos)
            if fixed is not None and self.pads:  # the rows past `n_tokens` are padding
                length = int(tokens.shape[1])
                real = UOp.variable(f"real_{length}", 1, length).bind(n_tokens)
                out = jit(tokens.contiguous(), following.contiguous(), self._temp, real, pos).realize()
            else:
                out = jit(tokens.contiguous(), following.contiguous(), self._temp, pos).realize()
            self._spec_prev, self._copy_prev = (self.pos + n_tokens - 1, 0), (False, False)
            self.pos += n_tokens
            return out
        if full or partial is not None:
            # what Transformer.__call__ does, with this session's own program for fixed lengths or partial chunks
            if self.model._position_count is None:
                self.model._position_count = 1
            pos = UOp.variable("start_pos_0", 0, self.max_context - 1).bind(self.pos)
            jit = partial[0] if partial is not None else fixed
            assert jit is not None
            out = jit(tokens.contiguous(), self._temp, pos).realize()
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
            result = self._sampled_ids = np.asarray(out.tolist(), dtype=np.int64).reshape(self.rows)
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
        ends: bool = False,
        inside_last: Callable[[int], None] | None = None,
    ) -> Any:
        """Feed `tokens` to every row, `chunk` tokens per call. Returns the output of the last call (see `step`).

        `before_last` is called before the last call is queued, with `pos` at its first token (where a caller
        takes a checkpoint of the state, `save_state`).

        `ends`: nothing the caller wants lies in the caches past these tokens. Where fixed-length programs take
        padding (`pads`), a last call of more than 256 tokens is then padded to the next of their lengths
        instead of being cut at the checkpoint's position: one call where there were two. Its padding rows write
        cache entries up to `padded_end`. The state at the start of its last 256 rows is kept, and
        `inside_last` is called with that position after the call is queued, in place of `before_last` (where the
        caller takes its checkpoint with `save_mid_state`).

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
        out, calls, first = None, 0, self.pos
        GlobalCounters.reset()
        start = time.perf_counter()
        # with a short-call program, the last call starts at the last multiple of small_chunk: the checkpoint taken
        # before it lies close to the end and on a grid every request shares, so a request that resumes there makes
        # the same calls as the one that took it
        small = self.small_chunk
        tail = (end - 1) // small * small if before_last is not None and small else 0
        self.padded_end = 0
        lengths = tuple(self._static) if ends and self.pads and self.pad_last else ()

        def call_at(pos: int) -> tuple[int, int]:
            return prefill_call(pos, end, self.chunk, tail, lengths, self.max_context)

        while self.pos < end:
            at = self.pos
            n, length = call_at(at)
            if before_last is not None and at + n == end and not length:
                before_last()
            self._feed_ple(tokens[at - first : at - first + n])
            if length > n and self._ple is not None:  # the padding rows read zeros: whatever they compute is finite
                self._ple.table.zero(at + n, length - n)
            self._prefetch_ple(tokens, first, at + n, call_at(at + n)[0])
            lo = self._v_pos.bind(at)
            # slicing by bound variables lets every chunk and every position reuse one compiled program;
            # with static chunks a full chunk has a fixed length and goes to its own program
            if length or n in self._static:
                count: Any = length or n
            else:
                partial = self._partial_for(n)
                count = (self._v_toks if partial is None else partial[1]).bind(n)
            following_rows = None if nxt is None else nxt[:, lo : lo + count]
            if wait_each:
                out = self._run("prefill", table[:, lo : lo + count], n, following_rows)
            else:
                out, calls = self._submit(table[:, lo : lo + count], n, following_rows), calls + 1
            if length:
                self.padded_end = at + length
                if inside_last is not None:
                    inside_last(at + length - SMALL_CHUNK)
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
            self._feed_ple(self._sampled_ids)
            return self._run("decode", self._sampled, 1)
        ids = [int(t) for t in tokens]
        if len(ids) != self.rows:
            raise ValueError(f"expected {self.rows} tokens, got {len(ids)}")
        if self.logits:
            return self._run_decode(ids)
        self._feed_ple(ids)
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
        self._feed_ple(ids[:1])
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
        self._feed_ple(ids)
        out = self.model.rollout_jit(self._decode_in, self._temp, self._v_pos.bind(self.pos))
        self.pos += 1
        result = self._read(out)
        self.trace.append(
            StepRecord("decode", self.rows, 1, time.perf_counter() - start, int(GlobalCounters.global_mem))
        )
        return result
