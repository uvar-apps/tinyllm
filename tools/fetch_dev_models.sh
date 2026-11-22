#!/usr/bin/env bash
# Downloads the small development models. Qwen3.5-0.8B has the same `qwen35`
# architecture as Qwen3.8-27B, so it exercises the same code paths in seconds.
#
#   tools/fetch_dev_models.sh              # into ~/.cache/models/Qwen3.5-0.8B-GGUF
#   TINYLLM_MODEL_DIR=/data/models tools/fetch_dev_models.sh
#   TINYLLM_DEV_MODELS="Qwen3.5-0.8B-Q8_0.gguf" tools/fetch_dev_models.sh   # a subset
set -euo pipefail

DIR="${TINYLLM_MODEL_DIR:-$HOME/.cache/models}/Qwen3.5-0.8B-GGUF"
BASE="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main"
# Q8_0: one quantization type, closest to the float model.
# UD-Q4_K_XL: the same mixed "dynamic" quantization style as the 27B reference file.
read -r -a FILES <<< "${TINYLLM_DEV_MODELS:-Qwen3.5-0.8B-Q8_0.gguf Qwen3.5-0.8B-UD-Q4_K_XL.gguf}"

mkdir -p "$DIR"
for f in "${FILES[@]}"; do
  if [ -s "$DIR/$f" ] && [ ! -e "$DIR/$f.part" ]; then
    echo "have $DIR/$f"
    continue
  fi
  echo "fetching $f"
  curl --fail --location --continue-at - --retry 3 --silent --show-error -o "$DIR/$f.part" "$BASE/$f"
  mv "$DIR/$f.part" "$DIR/$f"
done
ls -l "$DIR"
