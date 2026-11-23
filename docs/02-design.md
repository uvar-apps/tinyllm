# tinyllm — Design

Status: draft 1, 2026-11-22. Based on tinygrad commit `cdabdcb` (0.14.0, 2026-11-22).
Companion documents: [01-research-findings.md](01-research-findings.md) (evidence and measurements),
[03-implementation-plan.md](03-implementation-plan.md) (phases and exit criteria).

## 1. What we are building

tinyllm is an OpenAI-compatible LLM server for quantized GGUF models, built on tinygrad.
It aims to do on one workstation what vLLM does on datacenter GPUs: serve several concurrent
requests from one copy of the weights, reuse cached state between requests, and run the
hardware close to its physical limits.

The first target is the Qwen3.8 family (Qwen3.8-27B, then Qwen3.8-Flash-Next) on AMD Strix Halo
(Ryzen AI Max+ 395, Radeon 8060S, `gfx1151`) under Linux. The structure must let NVIDIA/CUDA,
other AMD GPUs and CPU be added later without rework.

**Goals, in priority order**

1. Correct, complete serving of Qwen3.8 GGUF files (mixed 4/5/6/8-bit quantization, BF16 tensors) through the OpenAI chat API, with streaming, tool calls and reasoning output.
2. Best achievable performance on Strix Halo, for one stream and for several concurrent streams.
3. A clean seam between serving logic and hardware-specific kernels, so other platforms are additive.
4. Over time, the most advanced execution techniques for quantized weights (section 4.7 and the plan's phase 4).

**Non-goals for now:** training or fine-tuning, multi-node serving, Windows/macOS, non-GGUF checkpoints, LoRA hot-swap.

## 2. The facts this design rests on

Each of these was verified on the target machine; numbers and sources are in the research document.

| # | Fact | Consequence |
|---|------|-------------|
| F1 | tinygrad already ships a GGUF LLM engine (`tinygrad/llm`) with hand-written AMD kernels that multiply directly on packed quantized weights. It runs on `gfx1151` through the kernel driver (KFD) with no changes. | We do not write a tensor engine or quantized kernels from scratch. |
| F2 | Single-stream decode of Qwen3.8-27B Q4 runs at 12.5 tok/s, reading 17.2 GB of weights per token at 214 GB/s. That is the machine's memory bandwidth. llama.cpp reaches the same number. | Single-stream decode cannot be made faster by better kernels. Only three things help: fewer bytes per token, more tokens per pass over the weights (speculative decoding), or more streams per pass (batching). |
| F3 | A decode step takes 80 ms for 1 stream, 90 ms for 4 and 106 ms for 8. Eight streams give 75 tok/s aggregate: 6 times the throughput for 33% more latency per token. | Continuous batching is the main throughput lever and must be the core of the engine. |
| F4 | tinygrad's server handles one request at a time, and its model class cannot run rows at different positions for Qwen3.8's recurrent layers. | Batching requires our own model graph and state management, not a wrapper. |
| F5 | Qwen3.8-27B (`qwen35` architecture) is a hybrid: 48 of 64 blocks are Gated DeltaNet recurrent layers, 16 are full attention. Recurrent state cannot be truncated the way a KV cache can. | Prefix reuse and speculative-decoding rollback need explicit state checkpoints. In exchange, KV memory is small: 64 KiB per token. |
| F6 | Warm prefill for one prompt runs at 191 tok/s in upstream's default chunks of 32 tokens and 290 tok/s in chunks of 128 (llama.cpp: 240 to 300); four prompts at once reach 337 tok/s. A pass costs about 100 ms plus 2 to 3 ms per token, and uses roughly a quarter of the GPU's theoretical FP16 throughput. | Chunk size is a first-order setting. Prefill still has kernel headroom, and long prompts make prefix caching essential. |
| F7 | Startup costs about 18 s to load weights and 45 s of JIT capture and kernel compilation, on every start. | Cold start needs a persistent program cache and a faster load path. |
| F8 | Qwen3.8 ships with drafting aids: a built-in multi-token-prediction (MTP) block in every GGUF, and a separate block-diffusion drafter (DFlash2). llama.cpp roughly doubles decode speed with MTP on this machine (24 to 26 tok/s). | Speculative decoding is required to match the best single-stream numbers. |

## 3. Key decisions

| # | Decision | Why | Rejected alternative |
|---|----------|-----|----------------------|
| D1 | **Layer on tinygrad; reuse its GGUF decoding and stateless quantized kernels; own the model graph, sequence state and everything above.** | The kernels are the hard, hardware-specific part and upstream improves them daily. The serving layer is what upstream lacks and, given its line-count discipline, will not grow. | (a) Thin wrapper over `tinygrad.llm`: cannot batch (F4), no logits access, no speculative decoding. (b) Clean-room engine: months of kernel work to reach parity with something that already exists. |
| D2 | **Pin tinygrad to an exact commit and touch it only through one shim package.** | `tinygrad/llm` changes daily and renames APIs without notice. | Tracking `master`: breaks constantly. Vendoring a fork: loses upstream kernel work. |
| D3 | **State lives in fixed "slots", with a host-side state store for parked sequences. No paged KV cache.** | Few concurrent sequences, large unified memory, and state copies are cheap here (a 30k-token KV cache copies in about 10 ms). Contiguous per-slot KV keeps the attention kernels simple and fast. | vLLM-style paged attention with block tables: solves fragmentation we do not have, and puts an indirection in the hottest attention kernels. Revisit for high-concurrency discrete GPUs. |
| D4 | **Batch as rectangular step programs in a few fixed sizes** (rows × tokens-per-row), JIT-compiled once each. | tinygrad's JIT wants static shapes; upstream kernels already have distinct fast paths for 1 to 8 rows and for multiples of 16. | Fully ragged token batches: best packing, but needs new kernels for every stateful op. Later optimisation. |
| D5 | **One process. The engine loop owns the main thread and the GPU; the HTTP server runs on an asyncio thread; they exchange plain messages.** | tinygrad devices and its JIT are not thread-safe, and its beam search needs the main thread. A message boundary lets the frontend move to its own process later. | Multi-process with ZMQ as in vLLM: more moving parts than a single-GPU box needs on day one. |
| D6 | **Logits leave the graph. A host-side sampler is the default; an in-graph sampler is a fast path.** | OpenAI-level sampling controls, logprobs, grammar masks and speculative verification all need logits. Reading logits back costs 3 ms per step with tinygrad's stock copy path (4.7 ms for 8 rows), 4% of an 80 ms step; phase 1 brings it under 1 ms. | Upstream's in-graph Gumbel-max only: temperature and nothing else. |
| D7 | **Speculative decoding is designed in from the start** (hidden-state taps, multi-position logits, recurrent-state replay), even though it lands in phase 3. | F2 and F8: it is the only way past the bandwidth ceiling for one stream, and retrofitting rollback into a recurrent model is painful. | Adding it later as a wrapper around `generate()`. |
| D8 | **Hardware-specific code is confined to "kernel packs" behind a small op interface.** | Makes CUDA and CPU additive; lets us replace upstream kernels one at a time with Strix-tuned ones. | Backend conditionals spread through model code, as upstream does. |

## 4. Architecture

### 4.1 Layers

```
 clients (OpenAI SDKs, agents, curl)
        │  HTTP + SSE
┌───────▼────────────────────────────────────────────────┐
│ server      OpenAI routes, auth, validation, SSE        │ asyncio thread
├─────────────────────────────────────────────────────────┤
│ frontend    chat template, tokenizer, tool/reasoning    │
│             parsing, incremental detokenizer, stops     │
├──────────────── EngineClient: message queues ───────────┤
│ engine      scheduler, sequences and slots, state store,│ main thread
│             sampler, speculative decoder, metrics       │ (owns the GPU)
├─────────────────────────────────────────────────────────┤
│ runner      step programs (decode / prefill / verify),  │
│             JIT and program cache                       │
├─────────────────────────────────────────────────────────┤
│ models      qwen35 (hybrid), qwen4exp (MoE), llama-like │
├─────────────────────────────────────────────────────────┤
│ ops         quant_linear, attention, gdn_scan, norms    │
│   packs:    amd_rdna │ generic │ (nv) │ (cpu_avx512)    │
├─────────────────────────────────────────────────────────┤
│ weights     GGUF parse, packed tensor store, repack     │
├─────────────────────────────────────────────────────────┤
│ tg shim     the only module that imports tinygrad       │
└─────────────────────────────────────────────────────────┘
   tinygrad (pinned): Tensor, UOp, JIT, AMD / NV / CPU runtimes
```

Rule: a layer imports only from layers below it. Nothing above `runner` knows what a tensor is; nothing above `ops` knows which GPU is present.

### 4.2 Request lifecycle

1. `server` validates the request and hands it to `frontend`.
2. `frontend` renders the chat template, tokenizes, and submits `EngineRequest(token_ids, sampling, limits, stream_handle)`.
3. `engine` matches the token prefix against live slots and the state store, assigns a slot, and restores the longest reusable state.
4. The scheduler interleaves this sequence's prefill chunks with other sequences' decode steps.
5. Each step returns logits for the rows that need them; the sampler picks tokens; tokens go back over the queue.
6. `frontend` detokenizes incrementally, routes text into `reasoning_content`, `content` and `tool_calls`, applies stop strings, and the server streams chunks.
7. On finish or client disconnect the sequence is parked: its state stays in its slot until the slot is needed, then moves to the state store.

### 4.3 Engine core

**Sequence.** A token list plus the state that corresponds to it. States: `queued`, `prefill`, `decode`, `parked`, `stored`.

**Slot.** One row of the working state tensors. The engine keeps active sequences in slots `0..R-1`, so a step program for `R` rows runs on a plain prefix view of the state tensors with no gather. When the active set changes, at most one sequence is moved to keep the prefix compact. Rows needed only to fill a bucket (3 active sequences in a 4-row program) run on a slot the engine treats as scratch.

**Scheduler.** A loop that picks one step per iteration:

- If sequences are ready to decode, batch all of them (up to the largest decode bucket) into one decode or verify step.
- If prefill work is waiting, run one prefill chunk. When both kinds of work exist they alternate under a token budget, so a 20,000-token prompt cannot stall running streams and a new request still gets a bounded time to first token.
- Admission: at most `slots` sequences are active; the rest wait in a FIFO queue with a length limit (HTTP 429 beyond it) and a per-request deadline.
- Cancellation is checked between steps; a disconnected client frees its row on the next step.
- Speculative decoding is used when few streams are active and switched off as the batch fills (section 4.10).

There is no preemption in the first version: slots are fixed-size, so a running sequence can never be evicted by memory pressure.

### 4.4 State and prefix reuse

Per slot, for Qwen3.8-27B:

| State | Shape | Size |
|-------|-------|------|
| KV cache, 16 attention layers | 2 × 4 KV heads × context × 256, fp16 | 64 KiB per token: 2 GiB at 32k context, 16 GiB at 262k |
| Recurrent state, 48 DeltaNet layers | 48 heads × 128 × 128, fp32 | 144 MiB |
| Convolution window, 48 DeltaNet layers | 3 × 10,240, fp32 | 6 MiB |

KV entries are addressed by position and never change once written, so any prefix of a KV cache is valid on its own. Recurrent state is a running summary and is only valid at the exact position it was taken.

**Checkpoints.** The engine copies the recurrent and convolution state (150 MiB, about 1.5 ms) at chosen positions: the end of the prompt, the end of each generated turn, and every few thousand tokens of a long prefill. A request whose tokens match a cached sequence up to position `p` resumes from the latest checkpoint at or before `p` and prefills only what follows. Without checkpoints, any edit to earlier conversation history would force a full re-prefill, which is what upstream does today.

**State store.** Host-side memory holding parked sequences (KV prefix plus checkpoints) under a byte budget with LRU eviction. It doubles as the prefix cache across slots: admitting a request copies the matching prefix into the assigned slot. On unified memory this is a plain memory copy at around 200 GB/s. Block-level deduplication between stored sequences is a later refinement.

**Resets** are engine operations (zero a slot's recurrent state) and not flags inside kernels. This is what lets rows in one step sit at unrelated positions.

### 4.5 Runner and step programs

A step program is a JIT-compiled function for a fixed shape:

```
step(kind, rows=R, tokens_per_row=T)
  inputs : tokens[R,T], pos[R] (bound variables), sampling inputs for the fast path
  state  : prefix views [0:R] of the slot state tensors, updated in place
  outputs: logits for the last token of each row, or for every position (verify),
           optional hidden-state taps at named layers
```

Initial buckets, chosen to match the kernel regimes measured on this machine:

| Kind | (R, T) buckets | Kernel regime |
|------|----------------|---------------|
| decode | (1,1) (2,1) (4,1) (8,1) | integer-dot on packed weights, one weight pass for all rows |
| verify | (1,8), later (2,4) | same, logits at every position |
| prefill | (1,128) for one prompt, (4,32) for concurrent prompts; a smaller chunk when decode rows are waiting, since a pass blocks them for its whole duration | FP16 matrix-multiply cores with in-kernel dequantization |

Programs are compiled on first use and kept. Compiled programs are pickled to disk, keyed by model file hash, tinygrad commit, device and bucket, so later starts skip JIT capture (F7).

`Runner` is an interface. Phase 1 ships `UpstreamRunner`, an adapter over `tinygrad.llm`'s model class with one stream, to get a working server immediately. Phase 2 adds `SlotRunner` with our own model graphs. The engine does not change between them.

### 4.6 Model definitions

Model graphs are written against `ops`, with state passed in explicitly instead of stored on layer objects. They are derived from upstream's `model.py` (MIT licence) and restricted to the architectures we serve, so each stays small.

`qwen35` (Qwen3.5, 3.6, 3.8 dense), per block:

- RMSNorm, then either Gated DeltaNet (3 of every 4 blocks) or grouped-query attention with QK-norm, RoPE and an output gate (every 4th block), then RMSNorm and a SwiGLU feed-forward.
- DeltaNet: `qkv` projection, causal depthwise convolution over the last 4 inputs, L2-normalised keys, a per-head decay and write strength, and the delta-rule state update done by one fused scan kernel over the tokens of a row.
- One extra block (`nextn`) is the MTP drafter; upstream skips it, we load it for section 4.10.

`qwen4exp` (Qwen3.8-Flash-Next) adds 512-expert MoE routing, hyper-connections, sparse attention through an indexer, compressed attention and hashed per-layer n-gram embeddings whose 100 GB table stays memory-mapped on the host. Upstream implements all of it for one stream; we port it to slots in phase 4.

A model definition also declares its **state layout** (what each slot holds, what a checkpoint must copy) and its **taps** (layers whose hidden states a drafter may request). The engine handles state generically from that declaration.

### 4.7 Ops, kernel packs and quantized execution

`ops` defines a handful of operations with reference semantics; a kernel pack supplies implementations and declares capabilities.

```
quant_linear(packed_weight, x[N, in]) -> [N, out]      stateless
attention(q, kv_state, pos[R], T)      -> out          reads and appends to the KV cache
gdn_scan(q, k, v, beta, alpha, state)  -> out          updates recurrent state in place
logits_head / rmsnorm / swiglu / conv_window
```

Packs:

- **`amd_rdna`** (first): wraps upstream's kernels for RDNA3/3.5/4. Stateless kernels are used as they are. The stateful ones (attention on the KV cache, DeltaNet scan) are adapted for per-row positions and no in-kernel reset. Strix-specific replacements are added here in phase 4.
- **`generic`**: plain tinygrad tensor operations on weights decoded once to fp16 at load. Correct on every tinygrad backend, memory-hungry, slow. It is the reference implementation, the CI backend, and the bring-up path for new platforms.
- **`nv`**, **`cpu_avx512`**: later, same interface.

What the `amd_rdna` pack does today, by regime:

| Rows per step | Method | Bound by |
|---------------|--------|----------|
| 1 to 8 | Activations quantized to int8 in groups of 32; packed weights unpacked in registers and combined with 4-way integer dot products; one pass over the weights serves every row | memory bandwidth |
| multiples of 16 | Weights dequantized to fp16 in registers and multiplied with 16×16×16 WMMA matrix cores | compute |
| small dense fp16/bf16 matrices | fp16 matrix-vector kernel | either |
| anything else | generic graph on lazily decoded weights | slow path |

Packed formats with native kernels: Q8_0, Q2_K to Q6_K, IQ1_S to IQ4_XS, IQ4_NL, MXFP4. Not covered: **Q4_0 and Q4_1**, and **float matrices with more than 2,048 outputs**, which take generic code generation. Those gaps hit two of the four priority formats. A plain Q4_0 file runs 89% of its weights on generic code. All of a BF16 file does, and so do the BF16 parts of mixed files: the output head of `UD-Q8_K_XL` reads at 9 GB/s and takes 62% of its decode step. Closing both gaps is phase 0B (research findings, section 5.5). Both gaps are closed by tinyllm's own kernels in `tinyllm.tg.kernels`, routed through a hook on tinygrad's `Linear` so they already serve the upstream engine of phases 0 and 1. `dense` reads BF16/F16 weights at the bandwidth limit for 1 to 8 rows, and `dense_mm` runs their prefill on the matrix cores. `q4` gives Q4_0/Q4_1 the integer-dot decode and matrix-core prefill paths that tinygrad has for its other formats. `q6k` and `q8` replace tinygrad's matrix-core prefill for Q6_K and Q8_0 with a loop that overlaps more loads (`wmma.py`) and a packed fp16 conversion; `q45k` and `iq4xs` do the same for Q4_K, Q5_K and IQ4_XS up to 32 rows. `TINYLLM_KERNELS` selects them.

The roadmap for "the most advanced algorithms for 8, 6 and 4-bit execution" follows directly from F2 and F6:

1. **More tokens per weight pass**: batching and speculative decoding (phases 2 and 3). Largest gains.
2. **Fewer kernel launches and less repeated work per pass**: fuse the gate and up projections of the feed-forward (same input, one activation quantization), fuse norm into the following projection, quantize shared inputs once for Q/K/V.
3. **Wider integer path**: extend the one-pass integer kernel from 8 to 16 rows (today 16 rows fall onto the compute-bound path and gain only 19% over 8); evaluate integer WMMA (int8 × int4/int8 matrix cores) for prefill so weights need not be expanded to fp16.
4. **Strix-specific tuning**: tile sizes, waves per workgroup, streaming loads and prefetch distance chosen for 40 compute units and shared DDR bandwidth, found by an autotuner and stored per device. Work that is sequential within a row (the DeltaNet scan, causal attention) costs about 16% on 128-token chunks; a chunk-parallel scan is a candidate.
5. **Load-time repacking**: rewrite GGUF blocks into layouts each kernel can load with fewer, wider reads; cached on disk next to the model.
6. **State bandwidth**: 8-bit KV cache for long contexts; fp16 recurrent state if accuracy holds.
7. **BF16**: native bf16 matrix cores where the hardware has them, exact conversion elsewhere.

### 4.8 Weights and loading

- GGUF parsing reuses `tinygrad.llm.gguf` behind the shim. The store keeps each tensor packed, with its quantization type and shape, and hands kernels zero-copy views.
- Mixed quantization per tensor is the normal case (the Unsloth "UD" files mix eight types), so kernel choice is per tensor.
- Load path target is NVMe speed, not the present 1 GB/s: parallel direct reads into device buffers first; mapping page-cache pages straight into the GPU address space on unified memory is a research item.
- Multi-part GGUF files and the separate `mtp-*.gguf` / drafter files are handled by the model resolver.

### 4.9 Sampling

Two paths behind one `SamplerPlan` built per request:

- **Host path (default).** Logits for the rows that need them are read to host memory; numpy applies logit bias, repetition/presence/frequency penalties, temperature, top-k, top-p, min-p, grammar masks, and returns token, logprobs and top-logprobs. Each request owns its RNG, so `seed` is reproducible regardless of what else is in the batch.
- **In-graph path.** For greedy or temperature/top-k only, sampling stays on the GPU and only token ids are read back.

Defaults come from the GGUF (`general.sampling.*`: temperature 1.0, top-k 20, top-p 0.95 for Qwen3.8) when the request does not set them.

### 4.10 Speculative decoding

Mechanism: a cheap drafter proposes `k` tokens; one verify step runs the target model on all of them in a single pass over the weights; the engine accepts the longest prefix the target agrees with, plus one corrected token. On this machine a pass over 8 positions of one sequence takes 108 ms against 80 ms for a single token, so accepted draft tokens are nearly free. With DFlash2's published acceptance of 4 to 5.5 tokens per verify step, that arithmetic gives 38 to 51 tok/s before drafter and rollback costs are added. Those acceptance figures are for an unquantized target, so expect less on Q4 weights.

Pieces:

- **`Drafter` interface**: `propose(sequence, k) -> tokens (+ probabilities)`. Implementations:
  - `MTPDrafter`: the model's own `nextn` block, applied repeatedly to the last hidden state and the embedding of the previous draft token. No extra file needed for the 27B.
  - `DFlashDrafter`: the separate 1.9 B-parameter block-diffusion model. It reads hidden states from five target layers and proposes a block of 7 tokens in one pass.
  - `NGramDrafter`: copies continuations from the prompt. Free, and effective for code edits and retrieval-heavy prompts.
- **Taps**: verify and decode programs can return hidden states at layers a drafter declares.
- **Acceptance**: exact match under greedy decoding; rejection sampling against the draft distribution under sampling, so output distribution is unchanged.
- **Rollback**: KV entries past the accepted length are simply ignored. Recurrent state is harder, because after a verify pass it has absorbed rejected tokens. The verify program therefore keeps each DeltaNet layer's small scan inputs (keys, values, gates per token). On partial acceptance the engine restores the pre-verify checkpoint and replays only the scan kernels over the accepted tokens. No weights are read again; the cost is a few milliseconds.
- **Policy**: draft length adapts to the recent acceptance rate. With many active streams the batch is already using the weight pass well, and published results show speculative gains disappearing at high concurrency, so the scheduler drops to plain batched decode.

Correctness rule: under greedy decoding, output with speculation on must be token-identical to output with it off. This is a standing test, because published MTP ports have shipped speedups that turned out to be corrupted text.

### 4.11 Frontend

- **Chat templates**: the Jinja template embedded in the GGUF, rendered in Jinja's immutable sandbox (model files are untrusted input), with `chat_template_kwargs` passed through (for example `enable_thinking`).
- **Tokenizer**: starts with upstream's pure-Python BPE behind a `Tokenizer` protocol, validated token-for-token against a reference on a multilingual and code corpus. A Rust-backed fast path built from the GGUF vocabulary follows if long prompts make tokenization measurable.
- **Output parsing**: streaming router for `<think>` reasoning blocks and tool calls (Qwen's JSON `<tool_call>` format first), adapted from upstream's server.
- **Detokenization**: incremental UTF-8 decoding and stop-string matching with hold-back, so partial stop sequences are never emitted.

### 4.12 API surface

| Milestone | Endpoints and features |
|-----------|------------------------|
| First | `POST /v1/chat/completions` and `/v1/completions` with streaming, usage, tools, reasoning content, stop, seed, max tokens, sampling parameters, logprobs; `GET /v1/models`; `GET /health` |
| Next | `/metrics` (Prometheus), `/tokenize`, `/detokenize`, `n > 1`, API keys |
| Later | `response_format` with JSON schema and grammars, `/v1/responses`, image input through the `mmproj` vision encoder, multiple loaded models |

### 4.13 Observability

- Prometheus metrics modelled on vLLM's: running and waiting requests, time to first token, inter-token latency, end-to-end latency, prompt and generated tokens, prefix-cache hit tokens, speculative acceptance rate, step time by kind and bucket, memory in use.
- A per-step trace (kind, rows, tokens, milliseconds, bytes read) that the benchmark harness consumes directly.
- Kernel-level profiling through tinygrad's profiler and AMD performance counters, wrapped in a `tinyllm profile` command.

### 4.14 Configuration

```
tinyllm serve  <model>  [--ctx 32768] [--slots 4] [--spec auto|mtp|dflash:<file>|ngram|off]
                        [--device auto|amd|nv|cpu] [--host] [--port] [--api-key]
tinyllm bench  <model>  [--prefill N] [--decode N] [--streams N] [--json out.json]
tinyllm inspect <model>           # architecture, tensor types, memory plan
tinyllm pull   <hf-repo>[:quant]  # download into the local model directory
```

`<model>` is a local path, a Hugging Face reference, or an alias from a small registry file.

## 5. Multi-platform strategy

| Platform | tinygrad device | Kernel pack | Status in this design |
|----------|-----------------|-------------|-----------------------|
| AMD RDNA3/3.5/4 on Linux (Strix Halo first) | `AMD` over KFD, compiler from ROCm (`libamd_comgr`) | `amd_rdna` | Primary. Needs the `amdgpu` kernel driver and ROCm's compiler library; no HIP runtime, hipBLAS or MIOpen. |
| AMD CDNA (MI300/MI350) | `AMD` | `amd_rdna` (upstream kernels already cover CDNA) | Expected to work; untested by us. |
| NVIDIA | `NV` or `CUDA` | `generic` first, then `nv` | Phase 6. Generic needs fp16 weights in memory, so native packed kernels are required before 27B is practical on 24 GB cards. |
| CPU (x86-64 AVX-512, later ARM) | `CPU` | `generic` first, then `cpu_avx512` | Phase 6. Also the CI backend from day one with a small model. |

What keeps this honest: the `generic` pack and a small `qwen35` model (Qwen3.5-0.8B) run in CI on CPU from phase 0, so model code can never quietly depend on AMD-only behaviour. Capability flags (packed-weight kernels, matrix cores, unified memory, bandwidth, memory size) are queried from the pack; the scheduler's bucket table and the memory planner read them instead of testing device names.

## 6. Memory plan on the target machine

123 GiB of RAM is shared by CPU and GPU; the GPU may map about 121 GiB.

| Item | Qwen3.8-27B Q4, 4 slots, 32k context | Same, 2 slots, 262k context |
|------|---------------------------------------|------------------------------|
| Weights | 17.2 GB | 17.2 GB |
| Slot state (KV + recurrent) | 4 × 2.2 GB | 2 × 16.2 GB |
| State store budget | 8 GB | 32 GB |
| Scratch, logits, drafter | 2 GB | 3 GB |
| **Total** | **about 36 GB** | **about 85 GB** |

The planner computes this table from the model's state layout at startup, refuses configurations that do not fit, and prints the plan. Qwen3.8-Flash-Next at Q4 needs about 111 GB, most of it a host-mapped embedding table, and will run with a small slot count.

## 7. Performance targets on Strix Halo (Qwen3.8-27B, UD-Q4_K_XL)

| Metric | Measured today (tinygrad CLI) | llama.cpp (published) | Target |
|--------|-------------------------------|------------------------|--------|
| Decode, 1 stream, no speculation | 12.5 tok/s | about 12.5 | 12.5 (the ceiling), with full sampling features |
| Decode, 4 streams aggregate | 44 tok/s (probe: all rows at the same position) | not published | at least 40 with independent requests |
| Decode, 8 streams aggregate | 75 tok/s (same probe) | not published | at least 68 with independent requests |
| Decode, 1 stream, speculative | not available | 24 to 26.5 (MTP, Vulkan) | at least 25 in phase 3; stretch 35 with DFlash2 |
| Prefill, one prompt | 191 tok/s in chunks of 32, 290 in chunks of 128 | 240 to 300 | at least 270 from phase 1; at least 350 in phase 4 |
| Follow-up turn, time to first token | full state reused only if history is unchanged | checkpoint-based | pay only for new tokens, including after edits |
| Start to ready, warm caches | about 63 s | a few seconds | under 15 s |

## 8. Testing and correctness

- **Oracle tests.** Upstream's single-stream engine is the oracle for our graphs: same weights, same kernels, greedy output must match token for token, and logits within a tight tolerance. Batched rows must match the same rows run alone.
- **Independent reference.** Logits compared against llama.cpp or Hugging Face Transformers on a small `qwen35` model, to catch errors shared with upstream.
- **Speculation.** Greedy output identical with speculation on and off, on a fixed prompt set, in CI.
- **Tokenizer and templates.** Golden files against reference tokenization and rendered prompts.
- **Scheduler.** A fake runner with scripted step times drives property tests for fairness, cancellation, deadline and admission behaviour without a GPU.
- **Performance gates.** The benchmark harness writes JSON; the Strix machine acts as a self-hosted runner that fails a change which regresses decode or prefill by more than a few percent.
- **Soak.** Hours of mixed traffic watching for memory growth, state corruption and GPU faults.

## 9. Security

- GGUF files are untrusted: bounds-checked header parsing, sandboxed templates, no pickle from model directories.
- The program cache is pickled; it lives in a user-owned cache directory and is keyed by content hashes.
- Server binds to localhost by default; API key and CORS are opt-in settings.
- Request limits: prompt tokens, generated tokens, queue length, body size.

## 10. Risks and open questions

| Risk | Mitigation |
|------|------------|
| Upstream churn in `tinygrad/llm` and the custom-kernel API (marked alpha) | Exact pin, single shim, oracle and performance suites run on every bump, bumps on a fixed cadence. |
| Upstream adds its own batching and diverges from ours | Good outcome for kernels; our model graphs stay thin so adopting upstream pieces is cheap. Contribute per-row attention upstream where welcome. |
| The integrated GPU also drives the display; a GPU fault can take the desktop down | Use only the kernel-driver interface (never tinygrad's userspace PCI driver on this machine), watchdog on hardware-exception events, soak tests before long unattended runs. |
| Recurrent-state replay for speculative rollback is more subtle than it looks | Prototype it first in phase 3 with the exact-match test before building drafters. |
| Python overhead per step grows with batch bookkeeping | Budget of 2 ms per step outside tinygrad, measured in the step trace. |
| Integer-quantized activations in the decode path differ slightly from llama.cpp's float path | First measurement in phase 0 (below): same top token at 95 of 96 positions on the 27B. Still to do: perplexity and logit divergence per quant type on a larger set; a float path for comparison. |
| tinygrad's default wait for GPU completion rounds any step longer than 200 ms up to the next tick of about 208 ms on this machine (found in phase 0) | The engine owns the completion wait and never calls the stock `synchronize()` on a long step. Benchmarks read the result back, which was verified against pure polling. Report upstream. |

### Questions answered in phase 0

**1. Which independent reference is available?** llama.cpp, built CPU-only from tag `b12629` by `tools/build_ref_llamacpp.sh` (the system install could not be repaired without adding Intel MKL). `tinyllm oracle reference` walks a stored greedy path and asks llama.cpp for its next-token distribution at each position.

| Record (tinygrad engine) | Same top token as llama.cpp | Mean / max log-probability difference | Tokenizer |
|---|---|---|---|
| Qwen3.5-0.8B Q8_0, CPU | 48 / 48 | 0.018 / 0.117 | identical |
| Qwen3.5-0.8B Q8_0, AMD | 96 / 96 | 0.015 / 0.198 | identical |
| Qwen3.5-0.8B UD-Q4_K_XL, AMD | 96 / 96 | 0.020 / 0.165 | identical |
| Qwen3.8-27B UD-Q4_K_XL, AMD | 95 / 96 | 0.011 / 0.144 | identical |

The one disagreement on the 27B is a near-tie (0.17 logits between the first and second candidates). AMD runs repeat bit-exactly. The CPU and AMD paths of tinygrad produce the same greedy tokens on the small model, with logits up to 0.44 apart, which is the footprint of the int8 activation quantization.

**2. Where does the warm-up go, and does a pickled JIT remove it?** It is Python graph rewriting, done twice per program (first run, then JIT capture). On the 27B, with tinygrad's compile cache warm, the compiler did not run at all; 62% of profiled time was in `tinygrad.uop`, 25% in C calls (hashing, cache lookups), under 2% in code generation. A pickled JIT skips all of it: on the small model, start to first token fell from 16.6 s to 1.4 s with identical output. The pickle embeds every buffer including the weights (835 MB for a 0.8 GB model), so large models need the weights bound from the GGUF at load instead. That is plan item 2f.

**3. Which HTTP stack?** FastAPI and uvicorn at the edge (decided 2026-11-23), on a thread of its own; the engine,
runner and frontend import no web code. Jinja2 renders chat templates in its immutable sandbox.

## 11. Repository layout

```
src/tinyllm/
  tg/          shim over the pinned tinygrad (devices, tensors, JIT, gguf, upstream kernels)
  weights/     GGUF store, model resolver, repack cache
  ops/         op interface, packs/{generic,amd_rdna}
  models/      qwen35.py, qwen4exp.py, state layouts, taps
  runner/      step programs, program cache, UpstreamRunner, SlotRunner
  engine/      scheduler, sequences, slots, state store, sampler, speculative/
  frontend/    templates, tokenizer, parsers, detokenizer
  server/      OpenAI routes, SSE, metrics
  cli.py
bench/         harness, probes, recorded baselines
tests/         unit, oracle, scheduler simulation, golden files
docs/
```
