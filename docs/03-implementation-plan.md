# tinyllm — Phased implementation plan

Status: draft 1, 2026-11-22. Read [02-design.md](02-design.md) first; section numbers below refer to it.

The first iteration (functionality and performance on Strix Halo) is phases 0 to 3.
Phases 4 to 6 are the "eventually" work: advanced quantized execution, platform features, other hardware.

Effort figures are rough estimates in focused engineer-weeks and exist only to show relative size.

## Overview

| Phase | Outcome | Size | Depends on |
|-------|---------|------|------------|
| 0 | Repository, pinned tinygrad, benchmark and oracle harness, recorded baselines | 1 | — |
| 1 | Single-stream OpenAI server on the upstream engine, at parity with the tinygrad CLI | 2 | 0 |
| 2 | Multi-sequence engine: continuous batching, slots, checkpoints, prefix reuse | 5 | 1 |
| 3 | Speculative decoding: n-gram, MTP, DFlash2 | 4 | 2 |
| 4 | Strix Halo kernel work, full quantization coverage, Qwen3.8-Flash-Next | 6+ | 2 (profiling can start during 3) |
| 5 | Platform features: structured output, vision, multi-model, more API | 4, parallelisable | 1 (some items need 2) |
| 6 | Other platforms: CPU, NVIDIA, other AMD GPUs | 6+ | 2, plus hardware |

Milestones: **M1** after phase 1 (a usable daily server), **M2** after phase 2 (several users or agents at once),
**M3** after phase 3 (fastest single-stream Qwen3.8-27B on this hardware).

Reference numbers for every exit criterion: Qwen3.8-27B `UD-Q4_K_XL`, 4096 context, performance power profile,
measured at 12.5 tok/s decode and 191 tok/s prefill with tinygrad's own settings (290 tok/s prefill in chunks of 128).
The recorded baselines are in `bench/baselines/`.

---

## Phase 0 — Foundations and baselines

Purpose: make every later claim measurable and every later change safe.

Work

1. `git init`, `pyproject.toml` managed with uv, `src/` layout from design section 11, ruff, mypy, pytest.
2. tinygrad pinned to commit `cdabdcb` as a git dependency; `tinyllm.tg` shim with device selection that forces the KFD interface on AMD and checks the pinned revision at import.
3. `tinyllm inspect`: GGUF header, per-tensor quantization histogram, state layout and memory plan (design section 6). The probe script used for the research is the starting point.
4. `tinyllm bench`: single-stream prefill and decode, batch-size and chunk-size sweeps, JSON output; today's numbers committed under `bench/baselines/`.
5. Development model: Qwen3.5-0.8B GGUF (same `qwen35` architecture as Qwen3.8-27B) for fast iteration and CI.
6. Oracle harness: run upstream `tinygrad.llm` and capture tokens and logits for a fixed prompt set.
7. Independent reference: repair the local llama.cpp build or use Hugging Face Transformers on CPU for the small model.
8. Measure where the 45 s warm-up goes (scheduling, rendering, compiling), and whether a pickled JIT removes it.
9. CI on CPU: lint, types, unit tests, small-model smoke test through the upstream engine.

Exit criteria

- `tinyllm bench` reproduces the baseline within 3% on the Strix machine.
- CI is green on a machine with no GPU.
- Open questions 1 and 2 of the design are answered in writing.

**Status: complete, 2026-11-22.**

| Exit criterion | Result |
|----------------|--------|
| `tinyllm bench` reproduces the baseline within 3% | Matches tinygrad's own benchmark script: decode 12.53 tok/s against 12.46 to 12.53, prefill 198.3 tok/s against 198.3. A repeat run against the recorded baseline was within 0.7% on every metric. |
| CI green without a GPU | The workflow is written but has not run on GitHub: there is no remote yet. Its commands pass locally with the device forced to the CPU: lint, types, 59 unit tests, and 6 tests that run the small model, including agreement with llama.cpp. |
| Open questions answered | Design section 10, "Questions answered in phase 0". |

What phase 0 found that changes later phases:

- **A measurement error in the research, now corrected.** tinygrad's stock wait rounds any GPU step longer than 200 ms up to the next 208 ms tick on this machine. Several prefill times in the first research draft were inflated by it, and two conclusions drawn from them were wrong. Decode figures were unaffected. The harness now waits by polling, and the engine must own its wait (phase 1, item 8).
- **Prefill is faster than first reported.** One prompt reaches 290 tok/s in chunks of 128 against 191 at upstream's default of 32. Phase 1 adopts the larger chunk; the phase 4 target moves from 300 to 350 tok/s.
- **Warm-up is Python, not compilation.** With tinygrad's compile cache warm the compiler never ran; a pickled JIT cut start to first token from 16.6 s to 1.4 s on the small model. Plan item 2f builds on that.
- **Reading logits to the host costs 3 ms per step**, more than assumed (phase 1, item 9).
- **Outputs agree with llama.cpp.** Same top token at 95 of 96 positions on the 27B and 96 of 96 on the small model in both quantizations, with identical tokenization.
- **The `UD-Q8_K_XL` file keeps its output head and attention query weights in BF16**, 2.3 B parameters that miss the fast kernels. `tinyllm inspect` reports this for any file; it sharpens phase 4d.

Not done in phase 0: a remote CI run, benchmarks of the Q5 and Q8 files, and a perplexity-scale accuracy measurement.

## Phase 1 — Single-stream server (vertical slice)

Purpose: a complete, correct server as early as possible, built on interfaces that survive phase 2.

Work

1. `Runner` interface and `UpstreamRunner`: adapts `tinygrad.llm`'s model for one sequence and returns logits instead of a sampled token.
2. Engine loop on the main thread, `EngineClient` message queues, request states, cancellation between steps, FIFO queue with limits and deadlines.
3. Frontend: sandboxed chat templates with `chat_template_kwargs`; tokenizer behind a protocol plus a validation corpus; streaming router for reasoning and tool calls; incremental detokenizer with stop strings.
4. Sampler: host path with temperature, top-k, top-p, min-p, repetition, presence and frequency penalties, logit bias, seed, logprobs; in-graph fast path for greedy.
5. Server: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`; SSE streaming; usage reporting including cached tokens.
6. Step trace and first metrics (time to first token, tokens per second, queue time).
7. Prefill in chunks of 128 tokens, queued without waiting between chunks (phase 0 measured 191 tok/s at upstream's 32 and 290 at 128).
8. Own the wait for GPU completion. tinygrad's stock wait rounds steps longer than 200 ms up to the next 208 ms tick on this machine; phase 0 switched the harness to pure polling. Decide between polling and a short sleep quantum, measure the CPU cost, keep fault detection, and report the behaviour upstream.
9. Cheap logits readback. The stock copy costs 3 ms per step (4.7 ms for 8 rows); return the top candidates from the device or map the buffer, and keep full logits for requests that need them.

Exit criteria

- OpenAI Python SDK scripts pass against Qwen3.8-27B: streaming chat, a tool-call round trip, reasoning on and off, stop sequences, seeded reproducibility.
- Decode at least 12.3 tok/s with the host sampler enabled (phase 0 measured 12.07 with the stock logits copy, 12.53 without).
- Prefill at least 270 tok/s for a single prompt.
- Tokenizer output identical to the reference on the validation corpus.
- Two simultaneous clients are served one after the other without errors; one-hour soak with no memory growth.

## Phase 2 — Multi-sequence engine

Purpose: the core of the product. Several independent requests share each pass over the weights.

Work

- **2a. Model graph.** `qwen35` written against `ops` with explicit state (design 4.6), running on the `generic` pack. Oracle test against upstream on the small model (CPU) and on the 27B (AMD).
- **2b. `amd_rdna` pack.** Wrap upstream's quantized linear kernels unchanged. DeltaNet scan without in-kernel reset. Attention with per-row positions: generic implementation first, then the flash kernels adapted to per-row valid lengths.
- **2c. `SlotRunner`.** Bucketed step programs (design 4.5), prefix-compact slots, scratch rows for partially filled buckets.
- **2d. Scheduler.** Continuous batching, prefill and decode interleaving under a token budget, admission control, deadlines.
- **2e. State reuse.** Checkpoints of recurrent state, the host-side state store with a byte budget, longest-prefix matching across live slots and stored sequences.
- **2f. Program cache.** Pickled JIT programs keyed by model hash, tinygrad commit, device and bucket, with weights and state bound from the loaded model instead of embedded in the pickle. Phase 0 showed the principle on the small model: start to first token fell from 16.6 s to 1.4 s, because the warm-up is Python graph rewriting, not compilation.

Exit criteria

- Four independent streams: at least 40 tok/s aggregate. Eight streams: at least 68 tok/s (90% of what the shared-position probe reached).
- One stream: within 3% of phase 1.
- A row decoded inside a batch is token-identical (greedy) to the same request run alone.
- Editing an earlier turn of a long conversation re-prefills only the tokens after the nearest checkpoint.
- A short new request arriving while another stream prefills 20,000 tokens gets its first token in under 1.5 s.
- Start to ready in under 25 s with a warm program cache.

## Phase 3 — Speculative decoding

Purpose: get past the bandwidth ceiling for a single stream (design 4.10).

Work

- **3a. Machinery first.** Verify programs with logits at every position, hidden-state taps, retention of DeltaNet scan inputs, checkpoint-and-replay rollback. Built and tested with a trivial drafter before any real one.
- **3b. `NGramDrafter`.** Prompt lookup; no model, exercises the whole path.
- **3c. `MTPDrafter`.** Load the `nextn` block from the main GGUF (and from separate `mtp-*.gguf` files where a model ships them that way); its own small KV cache.
- **3d. `DFlashDrafter`.** The `dflash` architecture: five non-causal blocks with dynamic convolutions, conditioned on five target layers, plus the candidate selector.
- **3e. Policy and sampling.** Rejection-sampling acceptance that preserves the output distribution; draft length adapted to acceptance rate; automatic fallback to plain batching as streams increase.

Exit criteria

- Greedy output is token-identical with speculation on and off across the test set. This test is permanent.
- One stream reaches at least 25 tok/s on code and JSON prompts (2× baseline, llama.cpp's published level); stretch goal 35 tok/s with DFlash2.
- No throughput regression with four or more streams.

## Phase 4 — Strix Halo optimisation and quantization coverage

Purpose: the advanced-execution work for 8, 6 and 4-bit weights and BF16 (design 4.7), driven by profiles.

Work

- **4a. Tooling.** `tinyllm profile` (per-kernel time and bytes, AMD performance counters) and a benchmark matrix across file types: `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q8_K_XL` (91% Q8_0, 9% BF16), plain `Q4_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`.
- **4b. Prefill.** Autotuned tile and workgroup parameters for 40 compute units; the roughly 100 ms fixed cost per pass; fused feed-forward; integer matrix-core experiment; chunk-parallel DeltaNet scan if profiles justify it.
- **4c. Decode.** Fused projections sharing one activation quantization; one-pass integer kernel extended to 16 rows.
- **4d. Dense F16/BF16.** A tuned kernel for large unquantized matrices; native bf16 where available.
- **4e. Startup.** Parallel direct-I/O weight loading; investigation of zero-copy mapping on unified memory; load-time repack cache.
- **4f. Long context.** 8-bit KV cache option; attention kernel behaviour beyond 100k tokens.
- **4g. Qwen3.8-Flash-Next.** The `qwen4exp` architecture (512-expert MoE, indexer, compressed attention, host-mapped n-gram embeddings) on the slot engine.
- **4h. Accuracy audit.** Perplexity and logit divergence of the integer-activation decode path against a float reference, per quantization type.

Exit criteria

- Prefill at least 350 tok/s for a single prompt.
- Decode reads weights at 95% or more of measured machine bandwidth for the Q4, Q5, Q6 and Q8 files; the BF16-heavy file within 10% of its own ceiling.
- Start to ready under 15 s.
- Flash-Next serves at least two concurrent streams.
- The accuracy report is published in `docs/`.

## Phase 5 — Platform features

Independent items; order by need. Each ships with conformance tests.

- Structured output: JSON schema and grammars through a logit-mask provider in the sampler (candidate libraries: xgrammar, llguidance).
- API: `/v1/responses`, `/tokenize`, `/detokenize`, `n > 1`, API keys, CORS.
- Vision input: the `mmproj` encoder (27-block vision transformer and merger) feeding image embeddings into prefill.
- Multiple models: load, unload and idle eviction within the memory plan.
- Operations: Prometheus dashboard, systemd unit, container image.

Exit criteria: schema-constrained output is valid on 100% of a schema suite at under 10% throughput cost; image descriptions match the reference implementation on a fixed image set.

## Phase 6 — Other platforms

- **6a. CPU.** Performance baseline of the `generic` pack; an AVX-512 VNNI pack for packed weights.
- **6b. NVIDIA.** `generic` pack bring-up on tinygrad's `NV`/`CUDA` devices; packed-weight kernels using integer dot products and tensor cores; memory planning for discrete VRAM.
- **6c. Other AMD GPUs.** RDNA4 and CDNA validation with per-device tuning tables.
- **6d. High concurrency.** Paged KV cache and larger buckets if discrete-GPU deployments call for them.

Exit criteria: the same oracle and API test suites pass on each platform, with a per-platform performance table against llama.cpp.

---

## Working practices across all phases

- Every tinygrad bump is its own change, gated by the oracle and performance suites.
- No performance claim without a committed benchmark JSON from the Strix machine.
- Kernel changes that are not Strix-specific are offered upstream.
- The integrated GPU drives the display: development and CI use the kernel-driver interface only.

## Immediate next steps (first working session)

1. Initialise the repository and tooling (phase 0, items 1 and 2).
2. Move the two research probes into `bench/` and turn them into `tinyllm inspect` and `tinyllm bench`.
3. Download Qwen3.5-0.8B and stand up the oracle harness.
4. Profile the warm-up to answer open question 2.

## Decisions wanted from you

1. **Approach.** Confirm decision D1: reuse tinygrad's kernels and GGUF code, own the model graph and serving. The alternative of a pure wrapper is faster to M1 but cannot batch.
2. **Order after phase 3.** Which comes first: Flash-Next support (4g), prefill speed (4b), structured output, or vision?
3. **HTTP stack.** FastAPI and uvicorn at the edge, or standard library only as tinygrad does?
4. **Other hardware.** Which NVIDIA or discrete AMD machine, if any, is available for phase 6.
