"""Sampling inside the decode program, for the common settings (plan, phase 1 item 9).

The host sampler needs the whole row of logits, 1 MB per token, and the copy plus the sampling cost about 1 ms of
host time per token at full CPU clock and three times that once the CPU clocks down under a sustained GPU load
(docs/01-research-findings.md, section 5.6). This sampler runs after the output head and returns one small
vector instead: the chosen token, the log-sum-exp of the row, and the CANDIDATES largest logits with their ids,
from which the host derives log-probabilities.

It applies, in the host sampler's order (`tinyllm.engine.sampling`): top-k (at most CANDIDATES), temperature,
top-p and min-p on the renormalized top-k distribution, then a draw with a uniform number the host passes in from
the request's own generator, so a seed reproduces. Temperature 0 with u = 0 picks the largest logit, lowest id
first, as argmax does. Penalties and logit bias need the whole row and stay on the host.
"""

from __future__ import annotations

from tinygrad import Tensor, dtypes

from tinyllm.tg.kernels.topk import topk

CANDIDATES = 32
PARAMS = 5  # temperature, top_k, top_p, min_p, u
OUTPUT = 2 + 2 * CANDIDATES  # token, log-sum-exp, candidate logits, candidate ids (as floats: ids < 2**24)


def sample_row(logits: Tensor, params: Tensor) -> Tensor:
    """`logits`: one row (any shape with V elements); `params`: (PARAMS,) float32. Returns (OUTPUT,) float32."""
    row = logits.reshape(-1).float()
    lse = row.logsumexp(0)
    vals, ids = topk(row, CANDIDATES)
    temperature, top_k, top_p, min_p, u = (params[i] for i in range(PARAMS))
    positions = Tensor.arange(CANDIDATES).to(row.device)
    scaled = (vals - vals[0]) / temperature.maximum(1e-6)
    p = scaled.exp() * (positions < top_k).float()
    p = p / p.sum()
    cum = p.cumsum()
    keep = ((cum - p) < top_p) & (p >= min_p * p[0])  # the smallest prefix reaching top_p; at least the first
    q = p * keep.float()
    q = q / q.sum()
    choice = (q.cumsum() <= u).cast(dtypes.int32).sum().minimum(CANDIDATES - 1)
    token = (positions == choice).where(ids, 0).sum()
    return Tensor.cat(token.float().reshape(1), lse.reshape(1), vals, ids.float())
