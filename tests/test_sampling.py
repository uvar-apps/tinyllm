"""The host sampler: each control on its own, seeds, and log-probabilities."""

from __future__ import annotations

import time

import numpy as np
import pytest

from tinyllm.engine.sampling import Sampler, SamplingParams


def _logits(values: dict[int, float], vocab: int = 32, floor: float = -10.0) -> np.ndarray:
    x = np.full(vocab, floor, dtype=np.float32)
    for token, value in values.items():
        x[token] = value
    return x


def _draws(params: SamplingParams, logits: np.ndarray, n: int = 400) -> list[int]:
    sampler = Sampler(params)
    out = []
    for _ in range(n):
        out.append(sampler.sample(logits).token)
        sampler.generated.clear()  # keep the distribution fixed between draws
        sampler.seen.clear()
    return out


def test_greedy_picks_the_largest_logit() -> None:
    sampler = Sampler(SamplingParams(temperature=0))
    assert sampler.sample(_logits({3: 2.0, 7: 5.0})).token == 7


def test_top_k_keeps_only_the_k_most_likely() -> None:
    draws = _draws(SamplingParams(top_k=2, seed=0), _logits({1: 3.0, 2: 2.9, 3: 2.8, 4: 2.7}))
    assert set(draws) == {1, 2}


def test_top_p_keeps_the_smallest_prefix_reaching_p() -> None:
    # probabilities about 0.5, 0.3, 0.2: top_p 0.7 needs the first two
    x = np.log(np.array([0.5, 0.3, 0.2], dtype=np.float32))
    logits = _logits({0: x[0], 1: x[1], 2: x[2]}, floor=-60)
    assert set(_draws(SamplingParams(top_p=0.7, seed=1), logits)) == {0, 1}
    assert set(_draws(SamplingParams(top_p=0.4, seed=1), logits)) == {0}


def test_top_p_after_top_k_uses_the_renormalized_distribution() -> None:
    # 0.4, 0.3, 0.2, 0.1: within the top 2 they become 0.57 and 0.43, so top_p 0.5 keeps one token
    # (over the whole vocabulary the first two would hold only 0.7, and 0.5 would need both)
    x = np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32))
    logits = _logits({i: x[i] for i in range(4)}, floor=-60)
    assert set(_draws(SamplingParams(top_k=2, top_p=0.5, seed=6), logits)) == {0}
    assert set(_draws(SamplingParams(top_p=0.5, seed=6), logits)) == {0, 1}


def test_min_p_drops_tokens_below_a_fraction_of_the_best() -> None:
    x = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float32))
    logits = _logits({0: x[0], 1: x[1], 2: x[2]}, floor=-60)
    assert set(_draws(SamplingParams(min_p=0.4, seed=2), logits)) == {0, 1}


def test_top_p_over_a_large_flat_vocabulary_grows_its_selection() -> None:
    # 1,000 equally likely tokens: top_p 0.5 must keep about half of them, beyond the first partial selection
    logits = np.zeros(248_320, dtype=np.float32) - 50
    logits[:1000] = 0
    draws = _draws(SamplingParams(top_p=0.5, seed=3), logits, n=2000)
    assert max(draws) < 1000 and len(set(draws)) > 300


def test_draws_follow_the_distribution() -> None:
    x = np.log(np.array([0.7, 0.3], dtype=np.float32))
    draws = _draws(SamplingParams(seed=4), _logits({0: x[0], 1: x[1]}, floor=-60), n=4000)
    assert abs(draws.count(0) / 4000 - 0.7) < 0.03


def test_temperature_sharpens_and_flattens() -> None:
    x = np.log(np.array([0.6, 0.4], dtype=np.float32))
    logits = _logits({0: x[0], 1: x[1]}, floor=-60)
    cold = _draws(SamplingParams(temperature=0.2, seed=5), logits, n=2000).count(0) / 2000
    hot = _draws(SamplingParams(temperature=5.0, seed=5), logits, n=2000).count(0) / 2000
    assert cold > 0.85 and 0.45 < hot < 0.6


def test_seed_reproduces_and_differs_across_seeds() -> None:
    logits = np.random.default_rng(0).standard_normal(1000).astype(np.float32)

    def run(seed: int) -> list[int]:
        sampler = Sampler(SamplingParams(seed=seed, top_k=50))
        return [sampler.sample(logits).token for _ in range(50)]

    assert run(7) == run(7) and run(7) != run(8)


def test_logit_bias_can_force_and_forbid() -> None:
    logits = _logits({3: 5.0, 4: 4.0})
    assert Sampler(SamplingParams(temperature=0, logit_bias={4: 100})).sample(logits).token == 4
    assert Sampler(SamplingParams(temperature=0, logit_bias={3: -100})).sample(logits).token == 4


def test_penalties() -> None:
    logits = _logits({3: 5.0, 4: 4.5})
    # repetition: 5.0 / 2 = 2.5 < 4.5 once token 3 was seen, also in the prompt
    assert Sampler(SamplingParams(temperature=0, repetition_penalty=2.0), prompt=[3]).sample(logits).token == 4
    # presence and frequency apply to generated tokens only
    assert Sampler(SamplingParams(temperature=0, presence_penalty=1.0), prompt=[3]).sample(logits).token == 3
    sampler = Sampler(SamplingParams(temperature=0, frequency_penalty=0.3))
    assert [sampler.sample(logits).token for _ in range(3)] == [3, 3, 4]  # 5.0 - 0.3 * 2 = 4.4 < 4.5


def test_logprobs_come_from_the_raw_distribution() -> None:
    x = np.log(np.array([0.5, 0.25, 0.25], dtype=np.float32))
    logits = _logits({0: x[0], 1: x[1], 2: x[2]}, floor=-80)
    got = Sampler(SamplingParams(temperature=0, logprobs=True, top_logprobs=2, logit_bias={1: 50})).sample(logits)
    assert got.token == 1  # the bias decides the token, but not the reported log-probabilities
    assert got.logprob == pytest.approx(np.log(0.25), abs=1e-5)
    assert [t for t, _ in got.top] == [0, 1] or [t for t, _ in got.top] == [0, 2]
    assert got.top[0][1] == pytest.approx(np.log(0.5), abs=1e-5)


def test_needs_logits_only_when_the_argmax_is_not_enough() -> None:
    assert not SamplingParams(temperature=0).needs_logits
    assert SamplingParams(temperature=0, logprobs=True).needs_logits
    assert SamplingParams(temperature=0, presence_penalty=0.5).needs_logits
    assert SamplingParams(temperature=0.7).needs_logits


def test_invalid_settings_are_rejected() -> None:
    for bad in ({"temperature": -1}, {"top_p": 0}, {"top_k": -1}, {"min_p": 2}, {"top_logprobs": 21}):
        with pytest.raises(ValueError):
            SamplingParams(**bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=1.0, top_k=20, top_p=0.95, seed=0, logprobs=True, top_logprobs=5),  # Qwen3.8's
        SamplingParams(temperature=1.0, seed=0),  # no filter at all: no sort of the vocabulary either
        SamplingParams(temperature=0.7, top_p=0.9, seed=0),
    ],
    ids=["qwen-defaults", "unfiltered", "top-p-only"],
)
def test_a_step_over_the_full_vocabulary_is_cheap(params: SamplingParams) -> None:
    """248,320 logits in a few milliseconds: a fraction of even the small model's 7 ms decode step."""
    logits = np.random.default_rng(1).standard_normal(248_320).astype(np.float32) * 3
    sampler = Sampler(params)
    sampler.sample(logits)
    start = time.perf_counter()
    for _ in range(20):
        sampler.sample(logits)
    assert (time.perf_counter() - start) / 20 < 0.004


def test_unfiltered_draws_follow_the_distribution() -> None:
    x = np.log(np.array([0.5, 0.3, 0.2], dtype=np.float32))
    draws = _draws(SamplingParams(seed=9), _logits({0: x[0], 1: x[1], 2: x[2]}, floor=-60), n=4000)
    assert abs(draws.count(0) / 4000 - 0.5) < 0.03 and abs(draws.count(2) / 4000 - 0.2) < 0.03
