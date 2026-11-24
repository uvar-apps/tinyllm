"""DFlash2's block convolution, selector walk and attention window on the CPU, against NumPy models of
llama.cpp's graph (`src/models/dflash2.cpp`) and driver (`common/speculative.cpp`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

DIM, GROUP, KERNEL, RANK, TOP_K, VOCAB, BLOCK = 64, 16, 2, 8, 4, 300, 8


def _kv(**over: Any) -> dict[str, Any]:
    a = "dflash"
    kv = {
        "general.architecture": a,
        f"{a}.embedding_length": DIM,
        f"{a}.block_count": 1,
        f"{a}.feed_forward_length": 96,
        f"{a}.attention.head_count": 4,
        f"{a}.attention.head_count_kv": 2,
        f"{a}.attention.key_length": 16,
        f"{a}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{a}.attention.causal": False,
        f"{a}.attention.sliding_window": 6,
        f"{a}.block_size": BLOCK,
        f"{a}.conv_kernel_size": KERNEL,
        f"{a}.conv_group_size": GROUP,
        f"{a}.selector_rank": RANK,
        f"{a}.selector_top_k": TOP_K,
        f"{a}.target_layers": [1, 2],
        f"{a}.rope.freq_base": 10000.0,
        "tokenizer.ggml.mask_token_id": VOCAB - 1,
    }
    kv.update(over)
    return kv


def _conv_reference(hidden: np.ndarray, dynamic: np.ndarray, base: np.ndarray, side: int) -> np.ndarray:
    """build_dflash2_conv for one block: hidden (T, D), dynamic (T, 2 * kernel * groups), base (2, kernel, D)."""
    T, D = hidden.shape
    coeff = dynamic.reshape(T, 2, KERNEL, D // GROUP)[:, side]
    out = np.zeros_like(hidden)
    for tap in range(min(KERNEL, T)):
        weight = np.repeat(coeff[:, tap], GROUP, axis=-1) + base[side, tap]  # channel c uses group c // GROUP
        previous = np.zeros_like(hidden)
        previous[tap:] = hidden[: T - tap]
        out += weight * previous
    return out


def test_block_convolution_matches_llama_cpp() -> None:
    from tinyllm.tg import Tensor
    from tinyllm.tg.dflash import DFlash

    rng = np.random.default_rng(0)
    drafter = DFlash(_kv(), 64)
    hidden = rng.standard_normal((BLOCK, DIM)).astype(np.float32)
    dynamic = rng.standard_normal((BLOCK, 2 * KERNEL * DIM // GROUP)).astype(np.float32)
    base = rng.standard_normal((2, KERNEL, DIM)).astype(np.float32)
    for side in (0, 1):
        got = drafter._conv(Tensor(hidden[None]), Tensor(dynamic[None]), Tensor(base), side).numpy()[0]
        np.testing.assert_allclose(got, _conv_reference(hidden, dynamic, base, side), rtol=1e-5, atol=1e-5)


def _walk_reference(
    hidden: np.ndarray, logits: np.ndarray, anchor: int, gate_w: np.ndarray, pred: np.ndarray, succ: np.ndarray
) -> list[int]:
    """The greedy selector walk of common_speculative_impl_draft_dflash2::draft."""
    drafts, chosen = [], 0
    previous_ids = np.asarray([anchor])
    for j in range(hidden.shape[0]):
        cands = np.argsort(-logits[j], kind="stable")[:TOP_K]
        gate = gate_w @ hidden[j]
        scores = (pred[previous_ids] * gate) @ succ[cands].T + logits[j, cands]  # (predecessors, successors)
        row = scores[0] if j == 0 else scores[chosen]
        chosen = int(np.argmax(row))
        drafts.append(int(cands[chosen]))
        previous_ids = cands
    return drafts


def test_selector_walk_matches_llama_cpp() -> None:
    from tinyllm.tg import Tensor
    from tinyllm.tg.dflash import DFlash

    rng = np.random.default_rng(1)
    drafter = DFlash(_kv(), 64)
    gate_w = rng.standard_normal((RANK, DIM)).astype(np.float32)
    pred = rng.standard_normal((VOCAB, RANK)).astype(np.float32)
    succ = rng.standard_normal((VOCAB, RANK)).astype(np.float32)
    drafter.selector_hidden.weight = Tensor(gate_w)
    drafter.selector_predecessor.weight, drafter.selector_successor.weight = Tensor(pred), Tensor(succ)
    for trial in range(5):
        hidden = rng.standard_normal((BLOCK - 1, DIM)).astype(np.float32)
        logits = (rng.standard_normal((BLOCK - 1, VOCAB)) * 3).astype(np.float32)
        anchor = int(rng.integers(VOCAB))
        got = drafter.select(Tensor(hidden), Tensor(logits), Tensor(np.asarray([anchor], np.int32))).numpy()
        assert got.tolist() == [_walk_reference(hidden, logits, anchor, gate_w, pred, succ)], trial


class _Target:
    """What the drafter borrows from the target model: the token embedding."""

    def __init__(self, rng: np.random.Generator) -> None:
        from tinyllm.tg import Tensor, nn

        self.token_embd = nn.Embedding(VOCAB, DIM)
        self.token_embd.weight = Tensor(rng.standard_normal((VOCAB, DIM)).astype(np.float32))


@pytest.mark.parametrize("changed", [2, 9])
def test_block_reads_the_window_and_nothing_after_it(changed: int) -> None:
    """A block at positions 10..17 with a window of 6 sees injected rows 5..9 (its first row's window) and its own
    rows; an injected row at 2 is outside every row's window, one at 9 inside."""
    from tinyllm.tg import Tensor, UOp, dtypes
    from tinyllm.tg.dflash import DFlash

    rng = np.random.default_rng(2)
    drafter, target = DFlash(_kv(), 64), _Target(rng)
    features = Tensor(rng.standard_normal((1, 10, 2 * DIM)).astype(np.float32))
    pos = UOp.variable("pos", 0, 63)
    stores = drafter.inject(features, pos.bind(0))  # allocates the cache on first use
    Tensor(drafter.cache[0].uop.after(*stores)).realize()
    block = Tensor(np.asarray([[5] + [VOCAB - 1] * (BLOCK - 1)], np.int32), dtype=dtypes.int32)
    before = drafter.hidden(target, block, pos.bind(10)).numpy()
    other = features.numpy().copy()
    other[0, changed] += 1.0
    stores = drafter.inject(Tensor(other), pos.bind(0))
    Tensor(drafter.cache[0].uop.after(*stores)).realize()
    after = drafter.hidden(target, block, pos.bind(10)).numpy()
    assert np.array_equal(before, after) == (changed == 2)
