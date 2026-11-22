# tinyllm — Research findings and options analysis

Status: 2026-11-22. Everything under "measured" was run on the target machine during this study.
Scripts used are in [`bench/probes/`](../bench/probes). The design that follows from this is in [02-design.md](02-design.md).

## 1. Summary

1. tinygrad at today's head already contains a capable GGUF LLM engine with AMD kernels that work on packed 1-to-8-bit weights. It runs unmodified on the Strix Halo GPU and already supports both Qwen3.8 architectures.
2. With it, Qwen3.8-27B at 4-bit decodes at 12.5 tok/s for one stream. That is the memory-bandwidth limit of the machine, and equals llama.cpp's plain decode speed.
3. Running several rows per step is nearly free: 8 rows decode at 75 tok/s in aggregate. tinygrad's own server does not exploit this; it serves one request at a time.
4. What is missing for a vLLM-like platform is everything above the kernels: batching of independent requests, state management for a recurrent model, speculative decoding, full sampling, a complete API.
5. Recommended approach: build tinyllm as layers on a pinned tinygrad, reusing its GGUF code and stateless kernels, and owning the model graph, state and serving.

## 2. Host platform (measured)

| Item | Value |
|------|-------|
| CPU | AMD Ryzen AI Max+ 395, 16 cores / 32 threads, AVX-512 |
| GPU | Radeon 8060S, PCI `1002:1586`, `gfx1151` (RDNA 3.5), 40 compute units, wave32 |
| Memory | 123 GiB unified; GPU-mappable (GTT) about 121 GiB; 2 GiB dedicated VRAM |
| OS | CachyOS, Linux 7.2.9, Mesa 26.2.4 (RADV), performance power profile |
| ROCm | `rocm-gfx1151-bin` 10.1.0 in `/opt/rocm` (provides `libamd_comgr`, which tinygrad's AMD compiler path needs) |
| Toolchain | Python 3.14.8 (GIL build), clang/LLVM 23.1.1, uv 0.13.5 |
| Other | A llama.cpp install exists but does not start (`libmkl_rt.so.3` missing); numpy is not installed system-wide |

The GPU also drives the display. tinygrad can reach an AMD GPU through the kernel driver (`KFD`) or through its own userspace driver, which unbinds the GPU from the kernel. All work here forced `DEV=KFD+AMD`. The userspace driver does not list this device ID, so an accidental fallback cannot take the display down, but the forced setting should stay.

## 3. tinygrad today (commit `cdabdcb`, version 0.14.0)

### 3.1 Runtimes

| tinygrad device | Relevance |
|-----------------|-----------|
| `AMD` over KFD, HIP renderer (compiles with ROCm's comgr) | **Works on `gfx1151`, verified.** The only path with packed-weight kernels. |
| `AMD` with LLVM renderer | Untested. Upstream's custom kernels require the HIP renderer. |
| `HIP`, `CL`, `WEBGPU` | Untested. Generic code generation only. |
| `NV` / `CUDA` | NVIDIA, generic code generation only for LLMs today. |
| `CPU` (clang or LLVM JIT) | Generic code generation only. Useful for CI. |

### 3.2 The `tinygrad/llm` module

About 3,000 lines, changing daily (five LLM commits in the five days up to this study).

- **GGUF**: parser with multi-part files; lazy decoding for F32, F16, BF16, Q4_0 to Q8_0, Q2_K to Q6_K, all IQ types, MXFP4.
- **Architectures**: llama-style, `qwen35` (Qwen3.5/3.6/3.8 dense), `qwen35moe`, `qwen4exp` (Qwen3.8-Flash-Next), DeepSeek, GLM, Kimi, gpt-oss, MiniMax and others, in one configurable model class.
- **AMD kernels** (RDNA3, RDNA4, CDNA; written as tinygrad UOps):
  - quantized linear on packed weights, integer-dot path: activations quantized to int8 per group of 32, weights unpacked in registers, `sudot4` dot products; up to 8 rows share one pass over the weights;
  - quantized linear, FP16 WMMA path for row counts that are multiples of 16, dequantizing in registers;
  - MoE expert kernel with in-kernel expert selection;
  - flash attention on an fp16 KV cache, separate decode and prefill kernels;
  - a fused Gated DeltaNet scan that updates recurrent state in place.
- **Server**: standard-library HTTP, `/v1/chat/completions` and `/v1/models`, streaming, reasoning and tool-call parsing for several formats, cancellation on disconnect, prefix reuse when a prompt extends the previous one.
- **Tokenizer**: pure-Python BPE from GGUF metadata, with a `qwen35` preset.

### 3.3 What it does not do

| Gap | Evidence |
|-----|----------|
| One request at a time | `serve.py`: "TODO: batch decodes across requests" |
| Rows at independent positions for recurrent layers | `GatedDeltaNetBlock` asserts a single shared position |
| Partial prefix reuse for hybrid models | State is reused only if the new prompt extends the cached one exactly; otherwise restart from token 0 |
| Sampling beyond temperature | Sampling is inside the graph (Gumbel-max); logits never leave it |
| Speculative decoding | The MTP block in the GGUF is skipped at load |
| Packed kernels off AMD | NVIDIA and CPU use generic graphs on lazily decoded weights |
| Fast start | 18 s load at 1 GB/s plus 45 s warm-up, every run |
| Vision input, structured output, metrics, completions endpoint | Absent |

### 3.4 Implication

The kernels are valuable and hard to reproduce; the serving layer is thin by design. tinygrad enforces a line budget and its custom-kernel API is marked alpha, so a serving platform belongs outside it, pinned to an exact commit.

## 4. Target models (from the GGUF files on this machine)

The official Qwen3.8 family on Hugging Face: `Qwen3.8-27B`, `Qwen3.8-Flash-Next`, and `Qwen3.8-2.4T-A95B` (far too large for this machine). There is no small Qwen3.8 model; Qwen3.5-0.8B has the same `qwen35` architecture and suits development and CI.

### 4.1 Qwen3.8-27B — architecture `qwen35`

| Property | Value |
|----------|-------|
| Blocks | 64, plus 1 `nextn` (MTP) block |
| Layer pattern | every 4th block is full attention (16); the other 48 are Gated DeltaNet |
| Width | 5,120; feed-forward 17,408; vocabulary 248,320 |
| Attention | 24 heads, 4 KV heads, head dimension 256, QK-norm, RoPE on 64 dimensions |
| DeltaNet | 48 value heads, 16 key heads, state 128 × 128 per head, convolution kernel 4 |
| Context | 262,144 |
| Default sampling in file | temperature 1.0, top-k 20, top-p 0.95 |

Local files and their tensor mix:

| File | Size | Average bits/weight | Composition by parameter count |
|------|------|---------------------|--------------------------------|
| `UD-Q4_K_XL` | 17.6 GB | 5.14 | Q5_K 42%, IQ4_XS 22%, Q4_K 20%, Q6_K 13%, IQ4_NL 2%, Q3_K 1%, Q8_0 0.5%, IQ3_S 0.3% |
| `UD-Q5_K_XL` | 20.9 GB | not inspected | — |
| `UD-Q8_K_XL` | 31.5 GB | 9.21 | Q8_0 91%, BF16 9% |

Eight quantization types in one file is normal for these "dynamic" quants, so per-tensor kernel selection is required. All eight have packed kernels upstream.

### 4.2 Qwen3.8-Flash-Next — architecture `qwen4exp`

48 blocks, width 2,560, 512 experts with 10 active per token plus a shared expert, hyper-connections (4 residual streams), sparse attention through an indexer (top-k 2,048), compressed attention on every 4th block, and hashed per-layer n-gram embeddings (16 tables of about 20 million rows). Same DeltaNet/attention interleave as the 27B. Local Q4 files total about 111 GB. Not run in this study.

### 4.3 Drafters and vision

| Artifact | What it is |
|----------|------------|
| `nextn` block inside the 27B GGUF | Multi-token prediction head: one extra attention block plus projection and norms, sharing the output head. llama.cpp uses it with `--spec-type draft-mtp`. |
| `mtp-Qwen3.8-Flash-Next-*.gguf` | The same idea shipped as separate files for Flash-Next. |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Architecture `dflash`: 1.9 B parameters, 5 non-causal blocks with dynamic convolutions, reads target layers 6, 20, 34, 48 and 62, drafts blocks of 8, with a low-rank candidate selector. 1.14 GB. |
| `mmproj-*.gguf` | Vision encoder: 27-block ViT, patch 16, image size 768, `qwen3vl_merger` projector to 5,120. 0.46 B parameters in BF16. |

## 5. Measurements on this machine

Model: Qwen3.8-27B `UD-Q4_K_XL`, context 4,096, tinygrad `cdabdcb`, `DEV=KFD+AMD`, greedy decoding.
Output was sane throughout (the benchmark prompt counts upward and the model continues the count).

### 5.1 Single stream (tinygrad's own CLI and benchmark script)

| Metric | Result |
|--------|--------|
| Decode | 80.3 ms per token = **12.46 tok/s**, reading 17.16 GB per token at **214 GB/s** |
| Decode at about 1,000 tokens of context | 11.8 tok/s |
| Prefill, warm, chunks of 32 | **198 tok/s** (1,024-token prompt) |
| Load | 17 to 18 s under the system Python 3.14 (1.0 GB/s, file already in page cache); 14 to 15 s in the project's Python 3.13 environment |
| Warm-up | 45 s under Python 3.14, 39.5 s under 3.13; not reduced by tinygrad's compile cache, because it is Python graph rewriting and not compilation (phase 0 profile) |

### 5.2 Rows per decode step (probe: all rows in one batch at the same position)

| Rows | Step time | Aggregate tok/s | Per stream tok/s |
|------|-----------|-----------------|------------------|
| 1 | 79.8 ms | 12.5 | 12.5 |
| 2 | 83.3 ms | 24.0 | 12.0 |
| 4 | 90.2 ms | 44.4 | 11.1 |
| 8 | 105.9 ms | 75.5 | 9.4 |
| 16 | 178.6 ms | 89.6 | 5.6 |

Up to 8 rows, step time fits roughly 76 ms + 3.7 ms per row. The fixed part is the pass over the weights; the per-row part is recurrent-state traffic (about 300 MB read and written per row), attention and the integer dot products.

At 16 rows the kernels switch from the integer-dot path to the FP16 WMMA path, the step becomes compute-bound (133 GB/s), and throughput rises only 19% for 69% more latency. Eight rows is the practical decode batch today. Extending the one-pass integer kernel to 16 rows would be worth about 118 tok/s if the linear fit holds.

### 5.3 Tokens per pass for one sequence and for prefill

| Shape (rows × tokens) | Pass time | Throughput | Note |
|-----------------------|-----------|------------|------|
| 1 × 8 | 107.6 ms | 74 tok/s | Integer-dot path. This is the cost of verifying 8 draft positions: 1.35 times a single-token step. |
| 1 × 16 | 157.9 ms | 101 tok/s | WMMA path. High fixed cost: 16 tokens take almost as long as 32. |
| 1 × 32 | 167 ms | 191 tok/s | WMMA path, upstream's default chunk. 198 tok/s when chunks are queued without waiting between them, as upstream's loop does. |
| 1 × 64 | 234 ms | 273 tok/s | |
| 1 × 128 | 441 ms | 290 tok/s | |
| 2 × 32 | 216 ms | 297 tok/s | |
| 4 × 32 | 380 ms | 337 tok/s | |
| 8 × 32 | 789 ms | 324 tok/s | |

Rows from 1 × 32 down are the phase 0 baseline recorded with `tinyllm bench` (`bench/baselines/`) and replace the figures first published here; see the correction below.

Two things stand out.

- **A pass has a fixed cost of roughly 100 ms, then 2 to 3 ms per token.** Upstream's default chunk of 32 tokens therefore leaves a third of the achievable prefill speed unused: a single prompt runs at 191 tok/s in chunks of 32 and 290 tok/s in chunks of 128, which is inside llama.cpp's published range.
- **One row of 128 tokens (441 ms) is 16% slower than four rows of 32 (380 ms)**, although the linear layers see the same 128 tokens. The difference is the work that is sequential within a row: the DeltaNet scan in 48 layers and causal attention.

**Correction.** The first version of this table reported 416 ms for 1 × 64, 2 × 32 and 4 × 32, 624 ms for 1 × 128, and 832 ms for 8 × 32, and drew two wrong conclusions from them: that a 64-token pass costs as much as a 128-token one, and that the DeltaNet scan dominates long single prompts. Those times were an artifact of how the probe waited for the GPU. tinygrad's `Device.synchronize()` polls for 200 ms and then sleeps in 200 ms steps waiting for a completion event that, on this machine, does not wake it; any step longer than 200 ms is rounded up to the next tick of about 208 ms. A controlled run of the same 2 × 32 call measured 416 ms with the default wait and 218 ms with pure polling. Every time below 200 ms in this document, including all decode steps up to 8 rows, was unaffected. The rounding is a real latency hazard for an engine, not only for benchmarks, and is carried into the design as a risk.

### 5.4 What the numbers mean

- **Decode is bandwidth-bound.** 17.16 GB × 12.46 per second is 214 GB/s. Published measurements of this platform's memory read bandwidth are about 239 GB/s, with 256 GB/s theoretical. No kernel can beat about 14 tok/s for this file one token at a time.
- **Therefore tokens per weight pass is the variable to maximise.** Eight rows cost 1.33 times one row. Eight positions of one row cost 1.35 times one position.
- **Prefill is compute-bound.** 190 to 340 tok/s corresponds to roughly 10 to 18 TFLOPS of useful FP16 work, against a community-estimated peak near 59 TFLOPS for 40 compute units.
- **State copies are cheap.** At these bandwidths a 150 MiB recurrent-state checkpoint copies in about 1.5 ms and a 2 GiB KV prefix in about 10 ms.

## 6. External baselines (published, not reproduced here)

llama.cpp, Qwen3.8-27B on Strix Halo:

| Configuration | Decode tok/s | Source |
|---------------|--------------|--------|
| Q4, no speculation, Vulkan or ROCm | about 12.5 | yandaq benchmark |
| Q4_K_M, MTP draft depth 4, Vulkan | 26.5 mean (23 prose, 29.5 code) | yandaq benchmark |
| UD-Q4_K_XL, MTP, Vulkan | 26.3 | yandaq benchmark |
| UD-Q8_K_XL, MTP, Vulkan | 15.2 | yandaq benchmark |
| Q4, MTP, ROCm | 18.7 | yandaq benchmark |
| Q8_0, MTP, ROCm | 12 to 22 depending on prompt | lilting.ch |
| Prefill, long prompts | 240 to 300 | yandaq benchmark |

DFlash2 for Qwen3.8-27B (vendor numbers, SGLang on one NVIDIA H200, 7 draft tokens per step, unquantized target):

| | MTP | DSpark | DFlash2 |
|---|-----|--------|---------|
| Acceptance length (five task sets) | 3.7 to 5.0 | 3.0 to 4.4 | 4.1 to 5.5 |
| Speedup over plain decode, 1 stream | 2.0 to 2.6× | 2.0 to 2.7× | 2.7 to 3.4× |
| Speedup at 32 concurrent streams | 0.8 to 1.0× | 0.7 to 1.1× | 1.0 to 1.5× |

Two cautions from that material. The DFlash authors recommend block sizes of 5 or less when target or drafter is quantized. And one public MTP port for Qwen reported speedups that were later traced to corrupted text, with the note that Gated DeltaNet state "is irreversible" and rollback needs snapshots or re-decoding.

Put together with section 5.3: llama.cpp's MTP result (2×) is the level to match, and a single-pass drafter such as DFlash2 should do better on a bandwidth-bound machine because drafting costs one pass over a 1.1 GB model instead of several passes over the MTP block and the 1 GB output head.

## 7. vLLM as the model

| vLLM concept | tinyllm position |
|--------------|------------------|
| Engine core loop with scheduler, separate from the API server | Adopt. Thread and message queue in one process instead of ZMQ between processes. |
| Continuous batching, re-formed every step | Adopt, with a few fixed batch shapes. |
| Chunked prefill under a token budget | Adopt. |
| PagedAttention and a KV block pool | Do not adopt now. Contiguous per-slot state plus a host-side store suits few sequences and unified memory. |
| Automatic prefix caching by block hash | Adapt: longest-prefix match on token lists plus recurrent-state checkpoints. |
| Speculative decoding (n-gram, draft model, MTP/EAGLE, DFlash) | Adopt; more valuable here than on datacenter GPUs. |
| Structured output through token masks | Adopt later through a mask hook in the sampler. |
| OpenAI-compatible server, Prometheus metrics | Adopt. |
| Tensor parallelism, LoRA | Not planned. tinygrad has sharding if it is ever needed. |

## 8. Options analysis

### 8.1 How to relate to `tinygrad.llm`

| Option | For | Against |
|--------|-----|---------|
| **A. Thin wrapper**: import the upstream model class, add a better server | Working in days; every upstream improvement is free | No batching of independent requests, no logits, no speculative decoding without patching upstream internals; breaks whenever upstream refactors |
| **B. Clean-room engine** on tinygrad's tensor core | Full control and a stable codebase | Re-creates thousands of lines of intricate kernels and a dozen quantization formats that already exist and are still improving |
| **C. Layered: reuse GGUF decoding and stateless kernels, own model graph, state and serving** | Batching, checkpoints and speculation designed in; upstream kernel work still flows in; hardware-specific code isolated for other platforms | We maintain our own `qwen35` graph and adapted stateful kernels; needs discipline around the pin |

**Recommendation: C**, entered through A. Phase 1 wraps the upstream model behind our runner interface to get a complete server quickly; phase 2 replaces the runner with our own graph without touching the layers above.

### 8.2 Other choices considered

| Topic | Choice | Reason |
|-------|--------|--------|
| AMD path | tinygrad `AMD` device over KFD | Verified; only path with packed kernels; does not need the HIP runtime |
| KV memory | Contiguous slots | See section 7 |
| Batch shapes | Fixed buckets 1, 2, 4, 8 rows | Matches the measured kernel regimes; 8 rows is the limit of the one-pass integer kernel |
| Sampling | On the host by default | Full feature set; reading logits is cheap on unified memory |
| Process model | One process, engine on the main thread | tinygrad is not thread-safe; a message boundary keeps the multi-process option open |
| Development model | Qwen3.5-0.8B | Same architecture as the 27B, fits CI |

## 9. Limits of this study

- Only the `UD-Q4_K_XL` file was benchmarked. The Q5 and Q8/BF16 files and Flash-Next were inspected but not run.
- Batch measurements used rows at a shared position, which upstream supports. Independent positions need the phase 2 engine and may cost a few percent more.
- No accuracy comparison against an independent implementation was made in this study. Phase 0 added a first one against llama.cpp (design document, "Questions answered in phase 0"); the effect of the integer-activation decode path still needs a perplexity-scale measurement.
- llama.cpp and DFlash numbers are from published sources with different software versions and settings; nothing was reproduced locally.
- CPU and NVIDIA backends were not exercised.

## 10. Sources

- tinygrad repository: https://github.com/tinygrad/tinygrad (commit `cdabdcb`)
- Qwen3.8 models: https://huggingface.co/Qwen/Qwen3.8-27B, https://huggingface.co/unsloth/Qwen3.8-27B-GGUF, https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
- Strix Halo llama.cpp benchmark: https://github.com/yandaq/qwen3.8-27b-strix-halo/blob/main/benchmark.md
- EVO-X2 ROCm and MTP test: https://lilting.ch/en/articles/qwen38-27b-evo-x2-rocm-mtp-test
- DFlash2 model card: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
- DFlash code and paper: https://github.com/z-lab/dflash, https://arxiv.org/abs/2602.06036
- MTP in llama.cpp: https://johnpaulwile.substack.com/p/multi-token-prediction-mtp-in-llamacpp
- MTP port post-mortem: https://github.com/quivent/qwen-mtp-research
- vLLM architecture: https://docs.vllm.ai/en/latest/design/arch_overview.html, https://vllm.ai/blog/anatomy-of-vllm
