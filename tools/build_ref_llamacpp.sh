#!/usr/bin/env bash
# Builds a CPU-only llama.cpp server as an independent reference implementation for
# logits and tokenization checks. It never touches the GPU.
#
#   tools/build_ref_llamacpp.sh            # builds the pinned tag into .ref/llama.cpp
#   LLAMACPP_TAG=b12700 tools/build_ref_llamacpp.sh
#
# The resulting binary is .ref/llama.cpp/build/bin/llama-server; point
# TINYLLM_REF_LLAMA_SERVER at another binary to use a different build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${TINYLLM_REF_DIR:-$ROOT/.ref}/llama.cpp"
TAG="${LLAMACPP_TAG:-b12675}"  # b12675: Q2_0 tensors (unsloth UD-IQ4_XS of Qwen3.8-Flash-Next)

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$DEST"
else
  git -C "$DEST" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
  git -C "$DEST" checkout --quiet "$TAG"
fi

cmake -S "$DEST" -B "$DEST/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON
cmake --build "$DEST/build" --target llama-server -j "$(nproc)"

echo "$TAG $(git -C "$DEST" rev-parse HEAD)" > "$DEST/build/REVISION"
echo "built $DEST/build/bin/llama-server ($(cat "$DEST/build/REVISION"))"
