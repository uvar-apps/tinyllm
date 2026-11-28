"""tinyllm's own kernels for tinygrad's AMD backend, and the hook that routes layers to them.

  dense     BF16 and F16 weights, decode and rows that are not a multiple of 16 (dense.py)
  dense_mm  BF16 and F16 weights, prefill on the matrix cores (densemm.py)
  q4        Q4_0 and Q4_1 weights (q4.py)
  q45k      Q4_K and Q5_K weights, prefill only; decode stays on tinygrad's kernel (q45k.py)
  iq4xs     IQ4_XS weights, prefill only; decode stays on tinygrad's kernel (iq4xs.py)
  q6k       Q6_K weights, prefill only; decode stays on tinygrad's kernel (q6k.py)
  q8        Q8_0 weights, prefill only; decode stays on tinygrad's kernel (q8.py)
  attention one-token attention on the matrix cores, and the speculative verify with it (decode_mma.py)
  moe       expert layers stored as Q5_0, Q5_1 or Q2_0 (moe_small.py)
  moe_mm    expert layers in prefill, grouped by expert on the matrix cores (moe_mm.py)
  qsa       Qwen3.8-Flash-Next's sparse attention: block keys, exact top-k selection, paged attention (qsa.py)
  hc        Qwen3.8-Flash-Next's gated residual in prefill: stream norm and inject gates in one kernel (hc.py)
  delta     the DeltaNet block in prefill: convolution, norms and output gate around its scan (deltafuse.py)

They follow the structure of tinygrad's kernels in `tinygrad/llm/kernels` and reuse its helpers,
so they are tied to the pinned tinygrad commit like everything else under `tinyllm.tg`.
"""

from tinyllm.tg.kernels.dense import dense_linear, find_dense
from tinyllm.tg.kernels.hook import ENV, KERNELS, install, installed, requested
from tinyllm.tg.kernels.q4 import find_q4, q4_linear

__all__ = ["ENV", "KERNELS", "dense_linear", "find_dense", "find_q4", "install", "installed", "q4_linear", "requested"]
