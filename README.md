# tinyllm

An OpenAI-compatible LLM server for quantized GGUF models, built on [tinygrad](https://github.com/tinygrad/tinygrad).
First target: the Qwen3.8 family on AMD Strix Halo (`gfx1151`) under Linux.

Status: **phase 0 complete** (foundations). There is no server yet. What exists is the tooling every later
phase is measured against: model inspection, a benchmark harness with recorded baselines, and an
oracle harness that checks outputs against tinygrad's own engine and against llama.cpp.

- [docs/01-research-findings.md](docs/01-research-findings.md): what was measured and why the design looks the way it does
- [docs/02-design.md](docs/02-design.md): architecture and decisions
- [docs/03-implementation-plan.md](docs/03-implementation-plan.md): phases, exit criteria and status

## Setup

Requires Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and `clang` on the path (tinygrad's CPU backend).
For the AMD GPU path: the `amdgpu` kernel driver with `/dev/kfd`, and ROCm's compiler library `libamd_comgr`.

```bash
uv sync
```

```bash
tools/fetch_dev_models.sh
```

```bash
tools/build_ref_llamacpp.sh
```

The first installs tinyllm with tinygrad pinned to an exact commit. The second downloads Qwen3.5-0.8B
(same architecture as Qwen3.8-27B) in two quantizations, 1.4 GB together. The third builds a CPU-only
llama.cpp used as an independent reference; it is optional.

## Commands

Describe a model file and the memory a serving configuration needs. Reads only the header, opens no device:

```bash
uv run tinyllm inspect Qwen3.8-27B-UD-Q4_K_XL.gguf --ctx 32768 --slots 4
```

Measure prefill and decode speed, one fresh process per configuration, and compare with the recorded baseline:

```bash
uv run tinyllm bench Qwen3.8-27B-UD-Q4_K_XL.gguf --rows 1,4,8 --compare bench/baselines/strix-halo.qwen3.8-27b-ud-q4_k_xl.json
```

Re-run a stored reference record and check that tokens and logits still match:

```bash
uv run tinyllm oracle check tests/golden/qwen35-0.8b-q8_0.amd.json
```

Compare a stored record with llama.cpp running the same file on the CPU:

```bash
uv run tinyllm oracle reference tests/golden/qwen35-0.8b-q8_0.amd.json
```

A model argument is a path, or a bare file name searched for under `~/.cache/models`
(override with `TINYLLM_MODEL_DIR`).

## Devices

`--device auto` (the default) picks the AMD GPU when a supported one is present and the CPU otherwise;
`uv run tinyllm device` shows the choice. On AMD, tinyllm always uses the kernel-driver interface
(`DEV=KFD+AMD`). tinygrad's userspace PCI driver unbinds the GPU from the kernel, which takes the display
down on a machine whose GPU also drives the screen; tinyllm refuses that configuration unless
`TINYLLM_ALLOW_AMD_USERSPACE=1` is set.

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

```bash
uv run pytest -m gpu
```

The first runs on the CPU: unit tests, plus the development model through tinygrad's engine and against
llama.cpp when those are installed (skipped otherwise). The second re-checks the stored GPU records and
needs the AMD GPU and the models they were captured from.

Layout: `src/tinyllm` (the package; only `tinyllm.tg` imports tinygrad), `tests` (with reference records in
`tests/golden`), `bench/baselines` (recorded performance), `tools` (build, fetch and one-off measurement scripts).
