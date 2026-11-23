# tinyllm — Research findings and options analysis

Status: 2026-11-22. Everything under "measured" was run on the target machine during this study.
Scripts used are in [`bench/probes/`](../bench/probes). The design that follows from this is in [02-design.md](02-design.md).

## 1. Summary

1. tinygrad at today's head already contains a capable GGUF LLM engine with AMD kernels that work on packed 1-to-8-bit weights in the K, IQ and Q8_0 formats. It runs unmodified on the Strix Halo GPU and already supports both Qwen3.8 architectures. It has no fast kernel for plain Q4_0/Q4_1 or for large BF16/F16 matrices (section 5.5).
2. With it, Qwen3.8-27B in Unsloth's `UD-Q4_K_XL` file decodes at 12.5 tok/s for one stream. That file is a mix averaging 5.14 bits per weight (42% Q5_K), not a uniform 4-bit file. The speed is the memory-bandwidth limit of the machine for that file size. Per-width results for 8, 6 and 4 bit and BF16 are in section 5.5.
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

### 5.5 Coverage of the priority weight widths: 8, 6 and 4 bit, and BF16

Added after phase 0. Sections 5.1 to 5.4 measured one mixed file only; this section covers each priority width.

**Which kernels each file reaches.** Measured with `tinyllm inspect`, from the header alone for files not yet downloaded (`tools/fetch_gguf_header.sh`, then `--header-only`). The share counts linear-layer parameters of the 64 main blocks; the extra MTP block, used only for speculative decoding, is reported separately.

| File | Size | Storage types | On tinygrad's fast kernels | With tinyllm's kernels |
|------|------|---------------|----------------------------|------------------------|
| unsloth `Q8_0` | 29.1 GB | 100% Q8_0 | 100% | 100% |
| bartowski `Q6_K` | 23.9 GB | 76% Q6_K, 22% Q8_0, MTP block in Q4_0 | 100% (MTP block: none) | 100% (MTP block too) |
| bartowski `Q4_K_M` | 17.4 GB | 75% Q4_K, 18% Q6_K, 5% Q8_0, MTP block in Q4_0 | 100% (MTP block: none) | 100% (MTP block too) |
| unsloth `Q4_0` | 16.1 GB | 87% Q4_0, 3% Q4_1 | **11%** | 100% |
| unsloth `BF16` | 54.7 GB | 100% BF16 | **0.8%** | 100% |
| unsloth `UD-Q8_K_XL` (local) | 31.5 GB | 91% Q8_0, 9% BF16 | 91% | 100% |
| unsloth `UD-Q4_K_XL` (local, sections 5.1 to 5.4) | 17.6 GB | 42% Q5_K, 22% IQ4_XS, 20% Q4_K, 13% Q6_K | 100% | 100% |

Two gaps in tinygrad's kernels follow directly: **Q4_0 and Q4_1 have no packed-weight kernel**, and **float matrices with more than 2,048 outputs** use generic code generation. The second covers all of BF16 and the BF16 parts of mixed files. tinyllm's own kernels (`src/tinyllm/tg/kernels/`, phase 0B) close both; the last column shows the result.

**What the BF16 gap costs.** `UD-Q8_K_XL`, one decode step, profiled per kernel with `tools/profile_step.py`:

| Kernel | Time | Share | Bandwidth |
|--------|------|-------|-----------|
| Output head, 248,320 × 5,120, BF16, generic | 272 ms | 62% | 9 GB/s |
| All Q8_0 weights, `linear_q8_0` × 448 | 111 ms | 25% | at the machine's limit |
| Attention query projections, BF16, generic × 16 | 50 ms | 11% | 43 GB/s |

The 8-bit kernels are not the problem. The file decodes at 2.0 tok/s for one row and 22.1 tok/s for eight (2.8 per row), because 2.3 B BF16 parameters are read far below bandwidth. Eight rows being faster per step than one (361 against 501 ms) shows the generic code picks a better schedule for eight rows than for one.

**First attempt: lifting a limit.** tinygrad already has a dense fp16 matrix-vector kernel, enabled only up to 2,048 outputs. Raising that limit in-process (`tools/dense_gemv_probe.py`; tinygrad itself unchanged) took one-row decode from 430.7 to 181.4 ms per token, 2.32 to 5.51 tok/s, with identical output tokens. On the all-BF16 file it gave only 0.47 to 1.39 tok/s, and the kernels touched 154 GB per step for a 55 GB file. The reason is that tinygrad's loader casts BF16 weights to fp16 lazily, so every step re-reads each weight, writes an fp16 copy and reads it again: 6 bytes moved per 2-byte weight.

**tinyllm's BF16/F16 decode kernel** (`src/tinyllm/tg/kernels/dense.py`, phase 0B). It reads the BF16 or F16 bytes straight from the model file's buffer, converts them exactly to fp32, and serves up to 8 rows of activations per read of the weights. Each workgroup copies its chunk of activations to shared memory once, which its 8 waves then share. A first version without that sharing reached 112 to 168 GB/s for one row and only 48 to 65 for eight, because every wave re-read 16 bytes of fp32 activations from cache per byte of weights. Per layer, Qwen3.8-27B shapes, BF16, weight bytes per second (`tools/dense_kernel_check.py`; relative error against NumPy below 3e-8 everywhere):

| Layer | tinygrad, 1 row | tinyllm, 1 row | tinyllm, 8 rows |
|-------|-----------------|----------------|-----------------|
| Output head, 248,320 × 5,120 | 11 GB/s (227 ms) | 222 GB/s (11.4 ms) | 216 GB/s |
| Feed-forward gate/up, 17,408 × 5,120 | 25 | 209 | 211 |
| Feed-forward down, 5,120 × 17,408 | 132 | 201 | 210 |
| Attention query, 12,288 × 5,120 | 45 | 194 | 206 |
| DeltaNet qkv, 10,240 × 5,120 | 42 | 201 | 201 |
| DeltaNet out, 5,120 × 6,144 | 65 | 185 | 175 |

F16 weights give the same speeds. Eight rows cost no more than one.

**tinyllm's Q4_0/Q4_1 kernels** (`src/tinyllm/tg/kernels/q4.py`, phase 0B). Both formats share IQ4_NL's block layout (an fp16 scale, for Q4_1 also an fp16 offset, then 32 4-bit codes), which tinygrad already handles at full speed, so these kernels reuse its machinery with their own dequantizer. Decode uses int8 activations and 4-way integer dot products; Q4_0 codes are mapped to signed bytes with a 16-entry byte-permute lookup. Prefill expands the weights to fp16 in registers for the matrix cores. Per layer, Q4_0, weight bytes per second including about 0.05 ms of fixed cost per measured call, which weighs on these small matrices (`tools/q4_kernel_check.py`; error against a NumPy model of each path's arithmetic below 4e-6 relative):

| Layer | tinygrad, 1 row | tinyllm, 1 row | tinyllm, 8 rows | tinygrad / tinyllm, 32 rows |
|-------|-----------------|----------------|-----------------|-----------------------------|
| Feed-forward gate/up, 17,408 × 5,120 | 112 GB/s | 176 | 163 | 33 / 96 |
| Feed-forward down, 5,120 × 17,408 | 56 | 182 | 164 | 29 / 112 |
| Attention query, 12,288 × 5,120 | 129 | 180 | 154 | 30 / 117 |
| DeltaNet qkv, 10,240 × 5,120 | 108 | 236 | 203 | 36 / 128 |
| DeltaNet out, 5,120 × 6,144 | 59 | 164 | 150 | 28 / 105 |

Q4_1 gives the same speeds.

**tinyllm's Q6_K prefill kernel** (`src/tinyllm/tg/kernels/q6k.py` and `wmma.py`, phase 0B). Q6_K decode was already at the bandwidth limit and stays on tinygrad's kernel; only the matrix-core prefill path is replaced. Finding what limits it took four measured steps, and the first two hypotheses were wrong:

1. *A cheaper conversion.* Building fp16 weights directly (OR a 6-bit code into the bit pattern of 1024.0, subtract 1056, multiply by the scale) instead of sign-extend, convert to fp32, scale, convert back. **No change** in speed.
2. *Fewer, wider loads.* Q6_K blocks are 210 bytes and only 2-byte aligned, so tinygrad reads them 2 bytes at a time. **No change** either: the compiled code showed LLVM had already merged those reads into 16-byte loads (`global_load_b128`).
3. *Speed of light.* With constant weights in place of the real ones (`tools/prefill_kernel_check.py --type q6_k --no-weights`) the kernel runs **2 to 6 times faster** (26 to 40 TFLOPS against 5 to 21). The weight path is the limit, but neither its arithmetic nor its instruction count was: tinygrad's loop handles one group of 32 weights per step and waits on that group's loads before the matrix-core work, so little memory traffic is in flight.
4. *Several groups per step* (`wmma.py`, a copy of tinygrad's loop with that one change) issues all their loads together: up to 1.6 times faster per layer. Once loads overlap, the conversion arithmetic does show (5 to 6 scalar instructions per weight in the machine code), and a packed form converting two weights per instruction (`v_perm_b32`, `v_or_b32`, `v_pk_add_f16`, `v_pk_mul_f16`) helps further at 32 rows.

Whole-model Q6_K prefill, one prompt, tok/s, for each configuration:

| Configuration | Chunk 32 | Chunk 128 |
|---------------|----------|-----------|
| tinygrad | 123.3 | 209.2 |
| 4 groups per step | 150.5 | **252.2** |
| 8 groups per step | 151.9 | 248.0 |
| 8 groups per step, packed conversion | **161.5** | 242.8 |

The kernel picks the best per row count: 8 groups packed up to 32 rows, 4 groups unpacked above. It rounds each block's scale to fp16 before multiplying, so a weight is rounded twice where tinygrad rounds once. Against the reference records captured with tinygrad's kernel, greedy tokens stay identical at all 96 positions, while logits move by up to 0.018 at the first generated token and up to 0.15 over 16 tokens. Agreement with llama.cpp is unchanged: 95 of 96 positions, mean log-probability difference 0.0102 against 0.0103 before. The records of the four files with Q6_K layers were therefore re-captured. Exact single rounding would keep the old records but give up the packed conversion, about 6% at chunk 32. The constant-weight ceiling is still 2 to 3 times higher, so more is available, probably by staging weight tiles in shared memory as llama.cpp's quantized matrix kernels do.


**tinyllm's Q8_0 prefill kernel** (`src/tinyllm/tg/kernels/q8.py`, phase 0B). tinygrad's Q8_0 matrix-core kernel took 85% of a 32-token prefill pass of the `Q8_0` file and 69% of the `UD-Q8_K_XL` one (the BF16 layers took another 19% of the latter). The same treatment as Q6_K, but the measurements came out differently. Packing the conversion two weights per instruction helps at every row count. More groups per step help only up to 32 rows and hurt above: 4 groups unpacked was up to 2 times slower at 128 rows. So for Q8_0 the conversion arithmetic is the larger cost, not memory latency. The kernel uses 2 groups per step packed up to 32 rows, and 1 group packed above. A Q8_0 code byte XOR 0x80 is q + 128, which OR-ed into fp16 1024.0 and less 1152 gives q exactly. The scale is already fp16, so each weight is rounded once, exactly as in tinygrad's path: the weights the matrix cores see are bit-identical. The stored reference records of `Q8_0`, `UD-Q8_K_XL` and the small Q8_0 model all match with a largest logit difference of 0.0000.

A lesson from checking that. In the Q6_K file, whose attention keys and values are Q8_0 while the queries are Q6_K, the outputs first changed although every Q8_0 layer was bit-identical in isolation. The kernel reshaped the activations before converting them to fp16, and tinygrad converts without reshaping. Layers reading the same input then got two differently shaped conversions, which changed how tinygrad fused and rounded the normalization that produces that input. Converting in exactly tinygrad's form made the model's output bit-identical again (largest difference 0.0000 with only the Q8_0 kernel on). Every tinyllm kernel that sits beside tinygrad's must take its input this way.

| Whole-model prefill, one prompt, tok/s | Chunk 32 | Chunk 128 |
|----------------------------------------|----------|-----------|
| `Q8_0`, tinygrad's kernel → tinyllm's | 130.8 → **175.2** | 239.9 → **273.8** |
| `UD-Q8_K_XL`, tinygrad's kernel → tinyllm's | 113.1 → **142.6** | 185.9 → **204.7** |

Decode is unchanged for both files. One layer shape is slightly slower than tinygrad's at 128 rows (feed-forward down, 1.31 against 1.25 ms per call); the rest are 7 to 37% faster.

**tinyllm's Q4_K, Q5_K and IQ4_XS prefill kernels** (`src/tinyllm/tg/kernels/q45k.py`, `iq4xs.py`, phase 0B). In a 32-token prefill of `UD-Q4_K_XL`, tinygrad's matrix-core kernels for these three types took 66% of the pass: Q5_K 33%, IQ4_XS 17%, Q4_K 15.5%. The same loop again, with four findings of its own:

1. *Loop structure matters as much as groups per step.* tinygrad's compiler splits a loop whose index appears as `i % 8` into an outer loop over 256-weight super-blocks and an inner one over its 8 groups. That moves the super-block's scale loads out of the inner loop. tinyllm's loop wrote the group index in a form the rule does not match, so with one group per step it was 1.4 to 1.7 times slower than tinygrad's kernel doing the same arithmetic. Written in the matching form (`wmma_linear(split_blocks=True)`), one group per step equals tinygrad's speed.
2. *Measure inside the model.* Per layer in isolation (`tools/prefill_kernel_check.py --configs`), a packed fp16 conversion with 4 groups per step looked best, also at 128 rows. Inside the model (`tools/profile_step.py --set`) several of those choices lose. The check tool reruns one layer, whose weights partly stay in the 32 MB last-level cache; in the model each layer's weights come from memory.
3. *Exact arithmetic costs little here.* Two groups per step with tinygrad's own conversion are within 3% of the best packed form for these layers. The packed forms (one `v_perm_b32` and one `v_pk_fma_f16` per two Q4_K or Q5_K weights; for IQ4_XS one `v_perm_b32` and one `v_pk_mul_f16` over tinygrad's 256-entry table in local memory) round a weight twice. The kernels default to the exact form, so their weights and outputs are bit-identical to tinygrad's; the packed form stays available as an option.
4. *Above 32 rows the matrix-core work itself is the limit.* With constant weights, a 128-row DeltaNet qkv call still takes 0.44 of its 0.64 ms, and changing the number of output tiles per wave does not move that floor (re-measured with an occupancy limit below: 0.345 ms of it is the matrix instructions, the rest activation loads). In the model, tinyllm's kernels were up to 10% slower than tinygrad's at 128 rows, so they leave those row counts to tinygrad's kernels.

Inside the model, ms per 32-token pass spent in each kernel, `UD-Q4_K_XL` with fixed-length chunks (below):

| Configuration | Q5_K | IQ4_XS | Q4_K |
|---------------|------|--------|------|
| tinygrad | 53.9 | 27.9 | 25.1 |
| 1 group per step, packed | 52.0 | 27.4 | 24.9 |
| **2 groups per step, tinygrad's arithmetic (default)** | **43.8** | **22.9** | **19.2** |
| 2 groups per step, packed | 42.3 | 22.4 | – |
| 4 groups per step, packed | 47.9 | 26.7 | 18.8 |
| 8 groups per step, packed | 45.9 | 30.8 | 19.9 |

**Fixed-length prefill chunks.** The same profile showed a generic elementwise kernel, `E_(p2*17408)`, taking 9% of a 32-token pass of `UD-Q4_K_XL`. It is the SiLU gate of the feed-forward blocks. `tinygrad.llm` compiles prefill once for a symbolic token count (1 to the chunk size), and tinygrad's scheduler gives an elementwise kernel over a symbolic axis no workgroup and no vector width, so this one ran about 7 times slower than memory allows. tinyllm's session can run full chunks through a second program compiled for exactly the chunk size, leaving only the last, partial chunk of a prompt to the symbolic one (`TINYLLM_STATIC_CHUNKS`, on by default on AMD). With it, generic code falls from 18% to 6% of a 32-token pass.

The two programs fuse and tile differently, so they round differently. Prompts shorter than one chunk are unaffected. For the two reference prompts that fill whole chunks, logits moved by up to 0.23 with the same greedy tokens (`tools/static_chunk_check.py` shows the same on the small model). Agreement with llama.cpp is unchanged: 95 of 96 positions, mean log-probability difference 0.0117 against 0.0115, with the same single near-tie. All reference records were re-captured with it on. The cost is a second compiled program: the first prompt whose length is not a multiple of the chunk compiles the variable-length program on first use, unless a warm-up has done so.

Whole-model prefill of `UD-Q4_K_XL`, one row, tok/s at chunk 32 (256-token prompt) / chunk 128 (1,024 tokens), all four combinations measured in one session (`tinyllm bench`; `TINYLLM_KERNELS=q6k,q8` gives the kernels as they were before this work):

| | Variable-length chunks | Fixed-length chunks |
|---|---|---|
| Previous kernels | 182.9 / 278.6 | 217.4 / 352.3 |
| New Q4_K, Q5_K and IQ4_XS kernels | 224.9 / 282.4 | **261.9 / 370.0** |

The two changes add up: each gives about a fifth more at chunk 32, while at chunk 128 only the fixed-length chunks help, since the new kernels leave those rows to tinygrad's. Against the recorded baseline (196.5 tok/s at chunk 32, 290.1 at chunk 128, measured on an earlier day) the gain is 33% and 28%. Fixed-length chunks speed up the 8-bit focus files as well: `Q8_0` 175.1 → 195.7 and 273.8 → 338.6, `UD-Q8_K_XL` 141.9 → 156.4 and 204.7 → 237.4. Decode is unchanged for all three.

**tinyllm's BF16/F16 prefill kernel, and an occupancy limit** (`src/tinyllm/tg/kernels/densemm.py`, phase 0B). Until now the BF16 decode kernel also served prefill, re-reading the weights for every 8 rows. The BF16 layers of `UD-Q8_K_XL` (its attention q, k and v projections) took 26% of a 32-token pass and 33% of a 128-token one, and the BF16 file prefilled at 34.5 tok/s. The new kernel runs multiples of 16 rows on tinyllm's matrix-core loop. It feeds the BF16 bits unchanged to the bf16 matrix instruction, so the activations are rounded to bf16 rather than fp16; llama.cpp's CPU path for BF16 weights does the same. Converting the weights to fp16 for the fp16 instruction instead is as fast up to 32 rows and 2.2 times slower at 128, where the conversion competes with the matrix instructions for the same vector units. It stays available as an option. Layers with fewer than 256 outputs stay on the decode kernel and its fp32 activations at every row count. These are the DeltaNet gate projections, whose outputs feed exponentials; in the small development model they are F16, and rounding their inputs to fp16 moved its logits by up to 0.31.

The first version reached only 60 to 175 GB/s at 16 and 32 rows, and changing the number of groups per step moved its speed by up to 2 times in no consistent direction. The compiled code showed why. The compiler schedules a kernel for the most waves per SIMD that its local-memory use allows. With tinygrad's small exchange buffer (4 KB at 16 rows) that is 16 waves, so it held the loop to 96 registers and issued each step's loads in three rounds, each costing a full memory latency. Whichever variant happened to get more registers was the fast one. Reserving local memory the kernel does not use lowers the target: with 64 KB per workgroup, one wave per SIMD, all of a step's loads go out at once. At 32 rows the compiled loop is the same with and without the reservation, apart from scheduling hints, yet the kernel is still 1.5 times faster with it. Fewer workgroups at a time read the weights faster; the reason is not established, and fewer concurrent streams of weight rows is the likely one. Per layer, ms per call, bf16 instruction, 2 groups per step, without → with the reservation (`tools/prefill_kernel_check.py --type bf16`):

| Layer | 16 rows | 32 rows | 128 rows |
|-------|---------|---------|----------|
| Feed-forward gate/up, 17,408 × 5,120 | 3.06 → 0.91 | 1.67 → **0.92** (195 GB/s) | 2.65 → 1.77 |
| Feed-forward down, 5,120 × 17,408 | 1.06 → 0.94 | 1.02 → 0.95 | 2.70 → 1.87 |
| Attention query, 12,288 × 5,120 | 1.86 → 0.68 | 1.02 → 0.69 | 2.06 → **1.31** (12.3 TFLOPS) |
| DeltaNet qkv, 10,240 × 5,120 | 1.48 → 0.59 | 0.83 → 0.57 | 1.82 → 1.13 |

tinygrad's own generated matrix multiply on the same bf16 data takes 1.79 ms for the attention query at 32 rows and 4.46 at 128.

The same lever applies to tinyllm's quantized kernels, since they share the loop, but the best amount differs by kernel and has to be found inside the model. For Q8_0, 24 KB is best. In a `UD-Q8_K_XL` pass that kernel's time falls from 136.4 to 125.6 ms at chunk 32 and from 313.9 to 297.3 at chunk 128, while 40 KB or more makes it slower (143.1 and 384.4). For the Q4_K, Q5_K and IQ4_XS kernels of `UD-Q4_K_XL`, reservations moved time between the kernels without shortening the pass by more than 1 ms, so they reserve nothing (`_config` in each module has the tables). The reservation changes no arithmetic, so the Q8_0 outputs stay bit-identical.

Whole-model prefill, one row, tok/s, against the recorded baselines (`tinyllm bench`; decode unchanged for every file):

| File | Chunk 32 | 8 rows × chunk 32 | Chunk 128 (1,024-token prompt) |
|------|----------|-------------------|--------------------------------|
| `BF16` | 34.5 → **117.4** | 34.7 → **232.0** | **233.2** (no earlier record) |
| `UD-Q8_K_XL` | 156.4 → **196.6** | 223.8 → **345.2** | 237.4 → **343.7** |
| `Q8_0` | 195.7 → **209.8** | 309.8 → **354.4** | 338.6 → **354.0** |

`UD-Q8_K_XL` now prefills as fast as `Q8_0` did before this step. BF16 at chunk 32 reads its 54.7 GB per pass at about 200 GB/s, so one 32-token chunk cannot go much faster on this machine; longer chunks or several rows are the way up, as the other two columns show.

**Prefill at 128 rows, re-measured with an occupancy limit.** The Q4_K and Q5_K finding above (the matrix-core loop itself is the limit above 32 rows) predates the occupancy lever, so it was measured again. It holds, and the loop's time now splits into parts. DeltaNet qkv, Q5_K, 128 rows, one call in isolation (`tools/prefill_kernel_check.py --no-weights --no-inputs`):

| Loop doing | ms | |
|------------|----|---|
| Matrix instructions only (constant weights and activations) | 0.345 | 39 TFLOPS |
| + activation loads (constant weights) | 0.44 | |
| + weight loads and conversion: tinygrad's kernel | 0.64 | 21 TFLOPS |
| tinyllm's loop, best setting (1 group, 16 KB reserved) | 0.675 | |

The weights alone (36 MB) could stream in about 0.17 ms, so a loop that overlapped everything would approach 0.35 to 0.4 ms. Reserving local memory cannot get there. Here the occupancy hides the activation loads, so every reservation made the constant-weight floor worse (0.44 to 0.64 ms at 64 KB), and no combination of reservation, groups per step and tiles beat tinygrad's kernel. Inside the model the same holds for the Q4_K, Q5_K, IQ4_XS and Q6_K kernels at chunk 128. For Q8_0, 128-token tiles with one output tile cut the kernel by 2 to 5% (300 to 285 to 293 ms per pass), but the pass shortened by only 0.6% as the kernels around it slowed, so tinygrad's tiles stay. What remains at 128 rows needs a different loop: activation tiles shared through local memory by all waves of a workgroup (now each wave loads its own copy of every fragment), and weight loads issued a step ahead of the matrix instructions that use them.

**That loop, built and measured: no gain.** Both mechanisms are in `wmma.py`, off by default and covered by a GPU test (`wmma_linear(share_inputs=True, prefetch=...)`; `tools/prefill_kernel_check.py --set q45k.SHARED=... --set q45k.PREFETCH=True`). Same layer, Q5_K DeltaNet qkv at 128 rows, where tinygrad's kernel takes 0.61 to 0.70 ms across runs:

| Loop | Best ms | |
|------|---------|---|
| Activations shared in local memory, one buffer (barriers before and after each step's tile) | 0.77 | the load latency is now paid inside every step |
| Two buffers, the next step's tile written after the current step's reads | 2.66 | the compiler issued the next step's loads at the end of the step, beside their stores, and put each local-memory read directly before its matrix instruction |
| Two buffers fed from registers loaded two steps ahead (the usual GEMM pipeline) | 0.64 | parity at best (64-token tile, 2 output tiles, 2 waves, packed conversion); every other shape slower |
| Weights loaded one step ahead into registers, converted the step after | 0.97 | 0.71 without; the loop compiled into 26 branches, which keeps loads and matrix work apart |
| Packed weight conversion in the existing loop | 0.69 | 0.70 unpacked: the conversion is not the cost either |

So at 128 rows neither the L2 traffic of the activations nor the conversion arithmetic is what costs. The remaining time is how the compiler orders loads, waits and matrix instructions, which tinygrad's path (C source, then LLVM) does not let a kernel control. Two ways forward remain: LLVM's scheduling hints for AMD GPUs (`__builtin_amdgcn_sched_group_barrier`, which fix the interleaving of memory and matrix instructions), or instruction-level code through tinygrad's own AMD assembler backend.

The Q8_0 kernel, the larger target (80% of a 128-token `UD-Q8_K_XL` pass, nearly all of `Q8_0`), behaves the same: at 128 rows the shared activations, the step-ahead weights and both are 6 to 30% slower than its plain loop (feed-forward gate/up 1.26 ms against 1.37 to 1.62). There the step-ahead weight loads do go out at the top of the step, but the activations are still loaded within it, so each step still waits. The scheduling hints were tried next (`wmma.SCHEDULE`, C placed at the end of the loop body). Every explicit interleaving of loads, vector and matrix instructions (`__builtin_amdgcn_sched_group_barrier`) was slower, up to 1.9 ms for that layer: the compiler's own order beats each imposed one. Its built-in strategy for matrix loops, `__builtin_amdgcn_iglp_opt(1)`, saves 1.5% in the Q8_0 kernel at 128 rows (297.5 to 292.3 ms per `UD-Q8_K_XL` pass, twice each) and nothing at 32; in the Q4_K and Q5_K kernels it was slower. The Q8_0 kernel now uses it above 32 rows: `tinyllm bench` at chunk 128, `Q8_0` 354.0 → **361.7** tok/s and `UD-Q8_K_XL` 343.7 → **348.6**, with 8 rows × 32 359.9 and 349.3; chunk 32 is unchanged. The outputs stay bit-identical.

So 128-row prefill sits at about 18 to 21 TFLOPS per kernel, and source-level restructuring and LLVM's hints are exhausted. What remains is instruction-level code (tinygrad's AMD assembler backend), a much larger project whose payoff the 0.345 ms matrix-only floor bounds at about 1.6 times for these calls.

**Speed of each width.** `tinyllm bench`, one row and eight rows, chunk 32, with tinyllm's kernels (arrows show the change from tinygrad's own paths; the Q6_K prefill kernel also speeds up the Q6_K layers of the other K files). Records are in `bench/baselines/`. The Q4_K_M, Q6_K and Q8_0 rows were re-recorded with nothing else running; the first recording, made while other files were downloading, was 1 to 4% lower. The three focus files (`Q8_0`, `UD-Q4_K_XL`, `UD-Q8_K_XL`) were re-recorded on 2026-11-23 with fixed-length prefill chunks, and `Q8_0`, `UD-Q8_K_XL` and `BF16` again later that day with the BF16 prefill kernel and the Q8_0 occupancy limit; the other rows predate both.

| File | Decode, 1 row | Bandwidth | Decode, 8 rows | Prefill, 1 row, chunk 32 |
|------|---------------|-----------|----------------|--------------------------|
| `Q4_0` (4-bit) | 4.62 → **13.58** tok/s | 75 → **215** GB/s | **78.8** tok/s | 9.0 → **190** tok/s |
| `Q4_K_M` (4-bit) | 12.42 tok/s | 214 GB/s | 75.8 tok/s | 181 → **193** tok/s |
| `Q6_K` (6-bit) | 9.22 tok/s | 215 GB/s | 58.3 tok/s | 123 → **162** tok/s |
| `Q8_0` (8-bit) | 7.97 tok/s | 222 GB/s | 52.2 tok/s | 131 → **210** tok/s |
| `BF16` | 0.47 → **4.23** tok/s | 25 → **220** GB/s | **31.8** tok/s | 1.2 → **117** tok/s |
| `UD-Q4_K_XL` (mixed, 5.14 bpw) | 12.52 tok/s | 215 GB/s | 75.6 tok/s | 188 → **262** tok/s |
| `UD-Q8_K_XL` (91% Q8_0, 9% BF16) | 2.00 → **7.36** tok/s | 62 → **222** GB/s | 22.1 → **49.1** tok/s | 18 → **197** tok/s |

What this says about the priority widths on gfx1151:

- **Decode of the K formats and Q8_0 is at the bandwidth limit** for 4, 6 and 8 bit alike, and 8 rows give 6.0 to 6.5 times one row. Single-row decode speed is set by file size, so it can only improve through fewer bytes per token or more tokens per weight pass (batching, speculative decoding).
- **Prefill is not at a common limit.** With tinygrad's kernels 6-bit and 8-bit prefill ran at 123 and 130 tok/s, against 181 for 4-bit. For Q6_K the cause turned out to be memory latency in tinygrad's matrix-core loop, not the dequantizer as first assumed (the Q6_K kernel paragraph above); tinyllm's kernel brings it to 162 tok/s. For Q8_0 the conversion arithmetic was the larger cost; tinyllm's kernel brings it to 175 tok/s, fixed-length prefill chunks to 196, and an occupancy limit to 210. For `UD-Q4_K_XL`, kernels for its Q4_K, Q5_K and IQ4_XS layers plus fixed-length chunks take it from 188 to 262 tok/s.
- **Q4_0 and BF16 decode now run at the bandwidth limit** with tinyllm's kernels; Q4_0 prefill (190 tok/s, matrix cores) is on par with Q4_K_M. BF16 prefill runs on the matrix cores at 117 tok/s for one 32-token chunk, which reads the 54.7 GB file at about 200 GB/s, and at 233 tok/s at chunk 128.

**Correctness of each width.** Greedy reference records captured on the GPU (`tests/golden/`) and compared with llama.cpp running the same file on the CPU (`tinyllm oracle reference`). All records were re-captured on 2026-11-23 with fixed-length prefill chunks and every tinyllm kernel on, and the `UD-Q8_K_XL` and `BF16` records again with the BF16 prefill kernel; the previous values are in brackets.

| File | Same top token as llama.cpp | Mean / max log-probability difference | Tokenizer |
|------|-----------------------------|----------------------------------------|-----------|
| `Q4_0`, with tinyllm's kernel | 95 / 96 | 0.0065 / 0.060 (0.007 / 0.062) | identical |
| `Q4_K_M` | 95 / 96 | 0.0094 / 0.110 (0.009 / 0.102) | identical |
| `Q6_K` | 95 / 96 | 0.0103 / 0.113 (0.010 / 0.113) | identical |
| `Q8_0` | 95 / 96 | 0.0049 / 0.064 (0.005 / 0.066) | identical |
| `UD-Q4_K_XL` | 95 / 96 | 0.0117 / 0.146 (0.0115) | identical |
| `UD-Q8_K_XL`, with tinyllm's kernels for its BF16 layers | 95 / 96 (96 / 96) | 0.0035 / 0.030 (0.0030 / 0.040) | identical |
| `BF16`, with tinyllm's kernels | 96 / 96 | 0.0010 / 0.010 (0.0010 / 0.010) | identical |

Each quantized file disagrees at one position, and each is a near-tie: the `json` prompt at step 6 in the K and Q8_0 files, and the `code` prompt at step 2 in `Q4_0` (0.02 logits between the first two candidates). `UD-Q8_K_XL` agreed there until its BF16 layers moved to the matrix cores, which round their activations to bf16 in prefill; it now picks the same token as `Q8_0`. The log-probability difference shrinks as the width grows, as expected if most of it comes from the 8-bit quantization of activations in tinygrad's decode kernels. tinyllm's BF16 decode kernel keeps activations in fp32 and converts BF16 weights exactly, and its BF16 results sit ten times closer to llama.cpp than any quantized file. Rounding prefill activations to bf16 did not change that: the `BF16` record kept every token, its logits moved by at most 0.02, and its agreement figures are unchanged.

### 5.6 Serving one stream (phase 1)

`tinyllm serve` runs the engine on the main thread and FastAPI on another (design 4.1). Decode samples on the host, so each step's input is a token chosen from the previous step's logits, and all host work per token adds to the GPU's 79.5 ms instead of overlapping it as in tinygrad's own loop. Three findings came out of getting that host work small.

**The CPU clocks down under a sustained GPU load.** Timing each part of a 27B decode step while sampling the CPU frequency showed the core running the engine drop from 4.7 to 1.5 GHz after about 80 steps (7 seconds) of continuous decoding. The Python part of a step then took three times as long: submitting one decode call went from 2.7 to 8.5 ms. The CPU and GPU share one power budget on this chip. The drop depends on how much the CPU does. With the host work cut down as described next, a bare decode loop kept the core at 3.8 to 4.7 GHz for 100 seconds, while the server, with sampling and its HTTP thread, still reached the low clock after about half a minute. It also means the decode figures recorded by `tinyllm bench`, which time 32 steps, are taken before the drop.

**Most host time per token was tinygrad's scheduler, not the model.** The upstream call path builds a new input tensor for every token, and realizing it walks every live tensor (`_apply_map_to_tensors`); reading the logits through `Tensor.numpy()` did the same again. The session now copies the token id into one realized buffer, replays the captured decode program directly, and copies the logits into one reused host array (`UpstreamSession._run_decode`). A bare decode step with logits readback went from 82.5 to 80.1 ms and stays at 80 to 81.6 ms over 1,200 steps: 12.26 to 12.49 tok/s. tinygrad's sample-mode loop does 12.52.

**Two smaller costs.** tinygrad's GPU wait polls in Python and holds the interpreter lock for the whole wait, so the HTTP thread competed with the engine on every token. `tinyllm.tg.set_wait("yield")` polls for 0.5 ms and then sleeps in 50 µs steps. And the first host sampler sorted the whole 248k-token vocabulary when no top-k was set, about 18 ms per token: the small model decoded at 40 tok/s with temperature 1 and at 139 greedy. Sampling now does top-k first, as Hugging Face, vLLM and llama.cpp do, and draws directly from the full distribution when nothing filters it: 0.4 to 1.4 ms per token.

Phase 1 exit checks on `UD-Q4_K_XL` through the OpenAI Python SDK (`tools/openai_sdk_check.py`, `tools/tokenizer_check.py`):

| Check | Result |
|-------|--------|
| Streaming chat, reasoning on and off, stop sequences, seeded reproducibility, tool-call round trip, two clients at once | all pass |
| Decode, 300 streamed tokens after the other checks, Qwen3.8's default sampling | **12.52** tok/s (target 12.3), sampled on the GPU (below); 11.95 with the host sampler |
| Prefill, one 6,412-token prompt, chunks of 128 | 317 tok/s (target 270) |
| Tokenizer against llama.cpp, 38 texts in many scripts, code, emoji, control tokens and a rendered chat prompt | identical |
| Soak, 55 minutes with the host sampler, 1,105 requests of five kinds (221 streams abandoned) | no failures; server memory 784.4 → 785.5 MB |
| Soak, 30 minutes with sampling on the GPU, 625 requests (125 abandoned) | no failures; server memory unchanged |

**Sampling on the GPU.** With the host sampler the decode target was missed by 3% under sustained load: about 1.5 ms of host work per token at full clock (submitting the call, copying 1 MB of logits, sampling), tripled once the CPU clocked down. The decode program now ends in a sampler (`tinyllm.tg.sampler`) for the common settings: greedy, or top-k up to 32 with temperature, top-p and min-p, the host passing in a uniform number from the request's own generator so that seeds still reproduce. It needs the 32 largest logits. tinygrad's `topk` sorts the whole row, 95 ms for 248,320 values, and taking the maximum k times costs 4.6 ms for k = 20; a bitonic sort in local memory in passes (each workgroup keeps the 32 largest of 1,024 values, three passes for the row) gives the exact result in 0.09 ms (`tinyllm.tg.kernels.topk`). The program returns 66 numbers: the token, the log-sum-exp of the row and the 32 candidates with their ids, from which the host derives log-probabilities. It matches a NumPy model of the same arithmetic on 240 of 240 draws (`tools/sampler_check.py`), and greedy decoding gives the same tokens as the host path.

The first version gained nothing (11.80 tok/s): its three copies per step (the token and the settings in, the result out) each cost 0.55 to 0.63 ms of host time in tinygrad's executor, while replaying the whole decode program cost 0.14 ms. Input and output now live in GPU memory mapped into the process (`BufferSpec(cpu_access=True)`), which the host writes and reads in place. A sampled decode step then takes 79.6 to 80.1 ms over 400 steps, 12.5 tok/s, as fast as tinygrad's own sample-mode loop, and the CPU stays at 5.1 GHz. End to end through the server: 12.52 tok/s, steady at 12.4 to 12.5 through every request of the check run. Requests with penalties, logit bias, no top-k or a top-k above 32 keep the host sampler. The cost is start-up: compiling the second decode program takes the 27B's from 79 to 94 seconds. A first version of the top-k kernel sorted blocks of 4,096 in two passes; it unrolled into a program so long that start-up took 141 seconds and the server held 5.6 GB of memory instead of 1.3, for a slower kernel (0.14 ms).

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
