#!/usr/bin/env bash
# Downloads only the start of a GGUF file from Hugging Face: enough for its metadata and
# tensor directory, so its quantization mix can be inspected before a full download.
#
#   tools/fetch_gguf_header.sh OUT_DIR REPO FILE [BYTES]
#   tools/fetch_gguf_header.sh /tmp/h bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-Q6_K.gguf
#
# Writes OUT_DIR/<repo>_<file>/head.gguf. Read it with bench/probes/gguf_info.py, which does not
# require the tensor data to be present (tinyllm inspect does, and rejects a truncated file).
set -euo pipefail

OUT="$1" REPO="$2" FILE="$3" BYTES="${4:-25165824}"
DIR="$OUT/$(echo "${REPO}_${FILE}" | tr '/' '_')"
mkdir -p "$DIR"
curl --fail --location --silent --show-error --max-time 180 --range "0-$((BYTES - 1))" \
  -o "$DIR/head.gguf" "https://huggingface.co/$REPO/resolve/main/$FILE"
echo "$DIR/head.gguf $(stat -c%s "$DIR/head.gguf")"
