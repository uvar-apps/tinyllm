# tinyllm

An OpenAI-compatible LLM server for quantized GGUF models, built on [tinygrad](https://github.com/tinygrad/tinygrad).
First target: the Qwen3.8 family on AMD Strix Halo (`gfx1151`) under Linux.

Status: **phase 1** (single-stream server) and **MTP speculative decoding** from phase 3. `tinyllm serve` answers
OpenAI's chat and completions APIs, with streaming, reasoning, tool calls, stop strings, seeds and logprobs, one
request at a time, drafting with the model's own multi-token-prediction block where the GGUF has one. Below it sit the tools
every phase is measured against: model inspection, a benchmark harness with recorded baselines, and an oracle
harness that checks outputs against tinygrad's own engine and against llama.cpp.

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

## Serving

```bash
uv run tinyllm serve Qwen3.8-27B-UD-Q4_K_XL.gguf --ctx 32768 --port 8000
```

It loads the model, compiles its programs (about 95 s for the 27B), and listens on
`http://127.0.0.1:8000/v1` (`--host 0.0.0.0` to accept other machines, `--api-key` to require a bearer token).
Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` (Prometheus).

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
reply = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3.8 thinks unless told not to
)
print(reply.choices[0].message.content)
```

- Chat prompts go through the model's own Jinja template, rendered in a sandbox. `chat_template_kwargs` and
  `reasoning_effort` pass through to it.
- Reasoning comes back as `reasoning_content`; tool calls as OpenAI `tool_calls`, with argument types taken from the
  request's tool schemas.
- Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`,
  `logit_bias`, `seed`, `logprobs` and `top_logprobs`. Unset values default to the model file's recommendation
  (Qwen3.8: temperature 1.0, top-k 20, top-p 0.95).
- One request runs at a time. Others wait, at most `--max-waiting` of them (16 by default); beyond that the server
  answers 429. A client that disconnects cancels its request.
- Prompt reuse: a request that extends the previous conversation reuses its state. One that departs from it, as a
  coding agent's next step does when the chat template re-renders the last reply without its reasoning, resumes from
  the nearest of up to eight copies of the recurrent state (`--state-checkpoints N`, 157 MB each on the 27B, 0 to
  turn them off). Copies are taken before each prompt's last prefill chunk and every 256 generated tokens. Usage
  reports the reused tokens as `prompt_tokens_details.cached_tokens`.
- On Qwen3.8-27B `UD-Q4_K_XL`, decode runs at 12.5 tok/s and a long prompt prefills at about 317 tok/s. Greedy
  decoding, and top-k up to 32 with temperature, top-p and min-p (Qwen3.8's defaults), sample inside the decode
  program on the GPU; penalties, logit bias and other top-k values sample on the host, at about 12 tok/s.

- Speculative decoding (`--speculative K`, default 3 when the GGUF has an MTP block, 0 turns it off): each step
  drafts K tokens with Qwen3.8's built-in MTP block and verifies them in one pass of the main model, which samples
  every position itself. `--draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf` drafts with DFlash2 instead, seven tokens
  per step from one pass of a 1.1 GB drafter: on the 27B about 50 tok/s on code and reasoning, 31 on average over
  mixed prompts, against 35 and 28 with the MTP block (research findings, sections 5.7 and 5.8). Output is the same as without speculation, token for token, greedy or seeded: the verify
  computes bit for bit what one-token decoding computes (`tools/rollback_check.py`). Drafts use a Q4_0 copy of the
  output head made at load (`TINYLLM_DRAFT_HEAD=output` uses the model's own; `q4:98304` keeps only the first
  98,304 tokens, faster for English and code, poor for other languages). Requests that sample on the host decode
  one token at a time. `/metrics` counts drafts and accepted drafts.

`tools/openai_sdk_check.py --base-url ...` runs the phase 1 checks against a running server; `tools/spec_check.py`
compares speculative and plain decoding on a model.

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

## Kernels

tinyllm adds its own GPU kernels where tinygrad has no fast path (`src/tinyllm/tg/kernels/`):

| Kernel | Weights | Qwen3.8-27B, tinygrad's path → tinyllm's |
|--------|---------|--------------------------------------------------|
| `dense` | BF16, F16 | BF16 file decode: 0.47 → 4.23 tok/s |
| `dense_mm` | BF16, F16, prefill (multiples of 16 rows) | BF16 file prefill: 34.5 → 117.4 tok/s; UD-Q8_K_XL: 156 → 197 tok/s |
| `q4` | Q4_0, Q4_1 | Q4_0 file decode: 4.62 → 13.58 tok/s |
| `q45k` | Q4_K, Q5_K, prefill up to 32 rows | UD-Q4_K_XL: 20% less time in these layers, output bit-identical to tinygrad's |
| `iq4xs` | IQ4_XS, prefill up to 32 rows | UD-Q4_K_XL: 18% less time in these layers, output bit-identical to tinygrad's |
| `q6k` | Q6_K, prefill only | Q6_K file prefill: 123 → 162 tok/s (decode was already at the limit) |
| `q8` | Q8_0, prefill only | Q8_0 file prefill: 131 → 210 tok/s (with fixed-length chunks, below) |
| `attention` | one-token attention and the speculative verify, gfx11 | DFlash2 cycle at 32k: 309 → 154 ms; at 128k: 404 → 283 ms |

They are on by default. `TINYLLM_KERNELS=none` (or a subset, such as `dense`) selects tinygrad's own paths for
comparison. `tools/dense_kernel_check.py`, `tools/q4_kernel_check.py` and `tools/prefill_kernel_check.py` check them on
the GPU; `tools/profile_step.py --set` compares kernel settings inside the model.

The matrix-core kernels share one loop (`wmma.py`). `dense_mm` and `q8` reserve local memory they do not use, which
lowers the occupancy the compiler schedules for; it then issues each step's loads at once instead of in rounds, and
the BF16 kernel runs up to 3.4 times faster per layer (`densemm.py` has the details).

On AMD, full prefill chunks also run through a program compiled for exactly the chunk size
(`TINYLLM_STATIC_CHUNKS=0` turns this off). With a symbolic token count tinygrad runs some elementwise kernels, such as
the feed-forward gate, several times slower. Prefill at chunk 32 / 128, tok/s, before fixed-length chunks, with
them, and with the BF16 prefill kernel, the Q8_0 occupancy limit and a scheduling hint as well:

| File | Before | Fixed-length chunks | Now |
|------|--------|---------------------|-----|
| `UD-Q4_K_XL` | 196.5 / 290.1 | 261.9 / 370.0 | 261.9 / 370.0 |
| `UD-Q8_K_XL` | 141.9 / 204.7 | 156.4 / 237.4 | 196.3 / 348.6 |
| `Q8_0` | 175.1 / 273.8 | 195.7 / 338.6 | 209.9 / 361.7 |
| `BF16` | 34.5 / – | – | 117.4 / 233.2 |

The cost is a second compiled program: the first prompt whose length is not a multiple of the chunk compiles the
variable-length one on first use.

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

```bash
uv run pytest -m gpu
```

The first runs on the CPU: unit tests, plus the development model through tinygrad's engine and against
llama.cpp when those are installed (skipped otherwise); the server's tests drive it through the OpenAI SDK with a
scripted model. The second re-checks the stored GPU records, checks the kernels, and serves the development model
through the SDK; it needs the AMD GPU and the models the records were captured from.

Layout: `src/tinyllm` (the package: `server`, `frontend`, `engine`, `runner`, `models`, `weights`, and `tg`, the only
part that imports tinygrad), `tests` (with reference records in `tests/golden`), `bench/baselines` (recorded
performance), `tools` (checks, build, fetch and one-off measurement scripts).
