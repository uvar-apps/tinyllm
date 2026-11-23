"""Pieces of speculative decoding that run on the CPU: the draft head's quantizer, the acceptance count, and the
sampler settings that keep speculation and one-token decoding drawing the same tokens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ggufgen import qwen35_kv, write_gguf

from tinyllm.engine.sampling import Sampler, SamplingParams


def _q4_0_reference(w: np.ndarray) -> bytes:
    """ggml's quantize_row_q4_0_ref."""
    out = bytearray()
    for block in w.astype(np.float32).reshape(-1, 32):
        i = int(np.argmax(np.abs(block)))
        d = np.float32(block[i]) / np.float32(-8)
        inv = np.float32(1) / d if d else np.float32(0)
        q = np.minimum(15, np.trunc(block * inv + np.float32(8.5))).astype(np.uint8)
        out += np.float16(d).tobytes() + (q[:16] | (q[16:] << 4)).astype(np.uint8).tobytes()
    return bytes(out)


def test_draft_head_quantizer_matches_ggml() -> None:
    from tinyllm.tg import Tensor
    from tinyllm.tg.mtp import quantize_q4_0

    rng = np.random.default_rng(0)
    w = (rng.standard_normal((40, 96)) * rng.uniform(0.01, 3, (40, 1))).astype(np.float16)
    w[3, :32] = 0  # an all-zero block
    raw = quantize_q4_0(Tensor(w), step=16).numpy()
    assert raw.dtype == np.uint16 and raw.size == 40 * 96 // 32 * 9
    assert raw.tobytes() == _q4_0_reference(w)


@pytest.mark.parametrize(
    ("sampled", "drafts", "expected"),
    [([5, 6, 7, 9], [5, 6, 7], 3), ([5, 6, 8, 9], [5, 6, 7], 2), ([4, 6, 7, 9], [5, 6, 7], 0), ([5, 0], [5], 1)],
)
def test_leading_matches(sampled: list[int], drafts: list[int], expected: int) -> None:
    from tinyllm.tg import Tensor
    from tinyllm.tg.speculative import leading_matches

    n = leading_matches(Tensor(np.asarray(sampled, np.float32)), Tensor(np.asarray(drafts, np.float32)))
    assert int(n.item()) == expected


def test_device_rows_depend_on_the_position_only() -> None:
    params = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=3)
    a, b = Sampler(params), Sampler(params)
    rows = a.device_rows(4)
    assert [r[:4] for r in rows] == [[0.8, 20.0, 0.9, 0.0]] * 4
    for row in rows[:2]:  # one-token decoding of the same positions draws the same numbers
        assert b.device_params() == row
        b.accept(0)
    assert b.device_rows(2) == rows[2:]
    assert len({r[4] for r in rows}) == 4 and all(0 <= r[4] < 1 for r in rows)
    assert Sampler(SamplingParams(temperature=0)).device_rows(2) == [[0.0, 1.0, 1.0, 0.0, 0.0]] * 2
    unseeded = Sampler(SamplingParams()), Sampler(SamplingParams())
    assert unseeded[0].device_rows(1) != unseeded[1].device_rows(1)


def test_serve_drafts_only_with_an_mtp_block(tmp_path: Path) -> None:
    from tinyllm.server.serve import has_mtp

    kv = qwen35_kv()
    assert has_mtp(write_gguf(tmp_path / "mtp.gguf", kv, []))
    kv["qwen35.nextn_predict_layers"] = 0
    assert not has_mtp(write_gguf(tmp_path / "plain.gguf", kv, []))
