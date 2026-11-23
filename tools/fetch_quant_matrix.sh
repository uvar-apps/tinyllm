#!/usr/bin/env bash
# Downloads the Qwen3.8-27B files used to benchmark each priority weight width (8, 6, 4 bit and BF16).
# The files were chosen for how uniform their storage types are (checked from their headers with
# tools/fetch_gguf_header.sh):
#
#   q4    bartowski Q4_K_M  17.4 GB   74.5% Q4_K, 18.4% Q6_K, 4.8% Q8_0
#   q4_0  unsloth   Q4_0    16.1 GB   86.9% Q4_0, 2.6% Q4_1, 5.5% Q5_K, 4.7% Q6_K
#   q6    bartowski Q6_K    23.9 GB   76.2% Q6_K, 22.1% Q8_0
#   q8    unsloth   Q8_0    29.1 GB   100% Q8_0
#   bf16  unsloth   BF16    54.7 GB   100% BF16 (two parts)
#
#   tools/fetch_quant_matrix.sh q4 q6 q8 bf16 q4_0   # any subset, in the order given
#
# Files land under $TINYLLM_MODEL_DIR (default ~/.cache/models)/<publisher>/Qwen3.8-27B-GGUF.
# Interrupted downloads resume.
set -euo pipefail

ROOT="${TINYLLM_MODEL_DIR:-$HOME/.cache/models}"
HF="${HF_CLI:-hf}"

for which in "$@"; do
  case "$which" in
    q4)   repo=bartowski/Qwen3.8-27B-GGUF; pattern="Qwen3.8-27B-Q4_K_M.gguf" ;;
    q4_0) repo=unsloth/Qwen3.8-27B-GGUF;   pattern="Qwen3.8-27B-Q4_0.gguf" ;;
    q6)   repo=bartowski/Qwen3.8-27B-GGUF; pattern="Qwen3.8-27B-Q6_K.gguf" ;;
    q8)   repo=unsloth/Qwen3.8-27B-GGUF;   pattern="Qwen3.8-27B-Q8_0.gguf" ;;
    bf16) repo=unsloth/Qwen3.8-27B-GGUF;   pattern="BF16/Qwen3.8-27B-BF16-*.gguf" ;;
    *) echo "unknown width '$which' (q4, q4_0, q6, q8, bf16)" >&2; exit 2 ;;
  esac
  dest="$ROOT/${repo%%/*}/Qwen3.8-27B-GGUF"
  echo "== $which: $repo $pattern -> $dest"
  "$HF" download "$repo" --include "$pattern" --local-dir "$dest"
done
