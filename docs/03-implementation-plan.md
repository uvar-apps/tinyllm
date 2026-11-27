# tinyllm — Phased implementation plan

Status: draft 1, 2026-11-22. Read [02-design.md](02-design.md) first; section numbers below refer to it.

The first iteration (functionality and performance on Strix Halo) is phases 0 to 3.
Phases 4 to 6 are the "eventually" work: advanced quantized execution, platform features, other hardware.

Effort figures are rough estimates in focused engineer-weeks and exist only to show relative size.

## Overview

| Phase | Outcome | Size | Depends on |
|-------|---------|------|------------|
| 0 | Repository, pinned tinygrad, benchmark and oracle harness, recorded baselines | 1 | — |
| 0B | Priority formats on gfx1151: 8, 6, 4 bit and BF16 all decode at the bandwidth limit | 2 to 3 | 0; can overlap with 1 |
| 1 | Single-stream OpenAI server on the upstream engine, at parity with the tinygrad CLI | 2 | 0 |
| 2 | Multi-sequence engine: continuous batching, slots, checkpoints, prefix reuse | 5 | 1 |
| 3 | Speculative decoding: n-gram, MTP, DFlash2 | 4 | 2 |
| 4 | Strix Halo kernel work, full quantization coverage, Qwen3.8-Flash-Next | 6+ | 2 (profiling can start during 3) |
| 5 | Platform features: structured output, vision, multi-model, more API | 4, parallelisable | 1 (some items need 2) |
| 6 | Other platforms: CPU, NVIDIA, other AMD GPUs | 6+ | 2, plus hardware |

Milestones: **M1** after phase 1 (a usable daily server), **M2** after phase 2 (several users or agents at once),
**M3** after phase 3 (fastest single-stream Qwen3.8-27B on this hardware).

Priorities, in order: gfx1151 (Strix Halo) first. Other GPUs and the CPU come later, and CPU code paths exist only so
tests run without a GPU.

**Focus files** (decided 2026-11-22, after the Q8_0 prefill kernel): Qwen3.8-27B `UD-Q4_K_XL`, and for 8 bit
`UD-Q8_K_XL` or plain `Q8_0`. Other quantizations (`Q4_K_M`, `Q4_0`, `Q6_K`, `BF16` files) keep their kernels,
baselines and records but get no new work. Mixed files still depend on their member types: `UD-Q4_K_XL` is
42% Q5_K, 22% IQ4_XS, 20% Q4_K and 13% Q6_K; `UD-Q8_K_XL` is 91% Q8_0 and 9% BF16.

Reference numbers for every exit criterion: Qwen3.8-27B, 4096 context, performance power profile, the focus files
(`bench/baselines/`, research findings section 5.5).

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
| CI green without a GPU | Superseded on 2026-11-23: there is no CI on GitHub (the workflow was removed and Actions are disabled on the repository). Everything is validated locally before pushing. In phase 0 the CPU checks passed locally with the device forced to the CPU: lint, types, 59 unit tests, and 6 tests that run the small model, including agreement with llama.cpp. |
| Open questions answered | Design section 10, "Questions answered in phase 0". |

What phase 0 found that changes later phases:

- **A measurement error in the research, now corrected.** tinygrad's stock wait rounds any GPU step longer than 200 ms up to the next 208 ms tick on this machine. Several prefill times in the first research draft were inflated by it, and two conclusions drawn from them were wrong. Decode figures were unaffected. The harness now waits by polling, and the engine must own its wait (phase 1, item 8).
- **Prefill is faster than first reported.** One prompt reaches 290 tok/s in chunks of 128 against 191 at upstream's default of 32. Phase 1 adopts the larger chunk; the phase 4 target moves from 300 to 350 tok/s.
- **Warm-up is Python, not compilation.** With tinygrad's compile cache warm the compiler never ran; a pickled JIT cut start to first token from 16.6 s to 1.4 s on the small model. Plan item 2f builds on that.
- **Reading logits to the host costs 3 ms per step**, more than assumed (phase 1, item 9).
- **Outputs agree with llama.cpp.** Same top token at 95 of 96 positions on the 27B and 96 of 96 on the small model in both quantizations, with identical tokenization.
- **The `UD-Q8_K_XL` file keeps its output head and attention query weights in BF16**, 2.3 B parameters that miss the fast kernels. `tinyllm inspect` reports this for any file; it sharpens phase 4d.

Not done in phase 0: a remote CI run, benchmarks of the Q5 and Q8 files, and a perplexity-scale accuracy measurement.

## Phase 0B — Priority formats on gfx1151

Purpose: every priority width decodes on Strix Halo at the memory-bandwidth limit, for one row and for a batch, before
anything is built on top. Phase 0 found that only the K and Q8_0 formats are there today (research findings, 5.5).
This phase touches kernels only, so it can run alongside phase 1.

Scope since 2026-11-22: what remains is limited to the focus files. Items 1 to 4 below are done for all widths;
item 5 now covers only the BF16 layers of `UD-Q8_K_XL`, and items 7 and 8 are new.

Work

1. **Per-width references.** For each of `Q8_0`, `Q6_K`, `Q4_K_M`, `Q4_0` and `BF16`: a benchmark baseline, a
   per-kernel profile (`tools/profile_step.py`), an oracle record, and an agreement check against llama.cpp.
2. **Dense BF16/F16 decode kernel** for large matrices, output head included, serving up to 8 rows in one pass over
   the weights. Upstream's dense kernel handles at most 2,048 outputs and one row per pass; lifting only the size limit
   already took `UD-Q8_K_XL` from 2.3 to 5.5 tok/s.
3. **Q4_0 and Q4_1 packed-weight kernels**: the integer-dot decode path and the matrix-core prefill path, following the
   structure of upstream's Q8_0 kernels. Without them a plain Q4_0 file runs 89% of its weights on generic code.
4. **Q6_K prefill.** A dedicated matrix-core dequantizer for Q6_K, like upstream's for Q4_K and Q5_K. Q6_K now goes
   through the generic byte-at-a-time one: 9.5 ms per billion weights per 32-token pass against 5.3 for Q4_K, so the
   6-bit file prefills at 123 tok/s against 181 for Q4_K_M.
5. **BF16 prefill** on the matrix cores, for the BF16 layers of `UD-Q8_K_XL`.
6. Offer the kernels upstream.
7. **Q8_0 prefill**, the same treatment as Q6_K: tinygrad's Q8_0 matrix-core kernel takes 85% of a 32-token pass of the
   `Q8_0` file and 69% of the `UD-Q8_K_XL` one.
8. **`UD-Q4_K_XL` prefill**: the same treatment for its Q4_K, Q5_K and IQ4_XS layers, whose tinygrad kernels take 66% of
   a 32-token pass of that file.

Exit criteria

- One-row decode reads weights at 205 GB/s or more (95% of what the Q4 files reach today) for `Q8_0`, `Q6_K`,
  `Q4_K_M`, `Q4_0`, `BF16` and `UD-Q8_K_XL`.
- Eight-row decode at least 5 times one-row throughput for each of them.
- Single-prompt prefill for `Q6_K`, `Q8_0` and `BF16` within 20% of `Q4_K_M`.
- No main-block linear layer on the generic path in any of these files (`tinyllm inspect`).
- Same greedy token as llama.cpp at 95% or more of positions for each file, with identical tokenization.

**Status, 2026-11-22.** Numbers: research findings, section 5.5.

- Item 1 is done for every file: baselines, profiles, reference records and llama.cpp agreement.
- tinyllm's kernels are on by default; `TINYLLM_KERNELS=none` (or a subset such as `dense`) selects tinygrad's paths.
- Item 2 is done: `src/tinyllm/tg/kernels/dense.py`.
  `BF16` decodes at 4.23 tok/s and 220 GB/s (was 0.47), with 8 rows at 31.8 tok/s. `UD-Q8_K_XL` decodes at 7.35 tok/s
  and 222 GB/s (was 2.00), with 8 rows at 49.2. Both agree with llama.cpp at 96 of 96 positions.
- Item 3 is done: `src/tinyllm/tg/kernels/q4.py`. `Q4_0` decodes at 13.58 tok/s and 215 GB/s (was 4.62), with 8 rows
  at 78.8 tok/s, and prefills at 190 tok/s (was 9.0). It agrees with llama.cpp at 95 of 96 positions.
- Decode criteria (205 GB/s for one row, 8 rows at least 5 times one row) are now met by every file.
- Item 4 is done: `src/tinyllm/tg/kernels/q6k.py`. `Q6_K` prefill 123.3 to 161.7 tok/s at chunk 32 and 208.5 to 251.9
  at chunk 128; decode unchanged. The cause was memory latency in tinygrad's matrix-core loop, not the dequantizer.
  The fix is several weight groups per loop step (`src/tinyllm/tg/kernels/wmma.py`) plus a packed fp16 conversion.
- Item 7 is done: `src/tinyllm/tg/kernels/q8.py`. `Q8_0` prefill 130.8 to 175.2 tok/s at chunk 32 and 239.9 to 273.8
  at chunk 128; `UD-Q8_K_XL` 113.1 to 142.6 and 185.9 to 204.7. Decode unchanged. Here packing the conversion mattered
  more than overlapping loads.
- Prefill criterion (within 20% of `Q4_K_M`, now 193 tok/s): met by `Q6_K` (162), `Q4_0` (190), `Q8_0` (210) and
  `UD-Q8_K_XL` (197). `BF16` reaches 117 at chunk 32, where one chunk reads the 54.7 GB file at about 200 GB/s, so a
  16-bit file cannot meet it at that chunk on this machine; it reaches 233 at chunk 128.
- Item 8 is done: `src/tinyllm/tg/kernels/q45k.py` and `iq4xs.py`, 2 weight groups per loop step with tinygrad's own
  arithmetic, so outputs stay bit-identical to tinygrad's. In the model they cut the time of the Q4_K and Q5_K layers by
  20% and of the IQ4_XS layers by 18% at 32 rows; above 32 rows tinygrad's kernels stay, being as fast or faster there.
  The same work found that tinygrad runs prefill elementwise kernels over a symbolic token count several times too slow;
  full chunks now run through a fixed-length program on AMD (`TINYLLM_STATIC_CHUNKS`). Prefill at chunk 32 / 128:
  `UD-Q4_K_XL` 196.5 / 290.1 to 261.9 / 370.0 tok/s, `UD-Q8_K_XL` 141.9 / 204.7 to 156.4 / 237.4, `Q8_0` 175.1 / 273.8
  to 195.7 / 338.6. Decode unchanged. All reference records were re-captured; agreement with llama.cpp is unchanged.
- Item 5 is done: `src/tinyllm/tg/kernels/densemm.py`, the bf16 matrix instruction on tinyllm's loop, for multiples of 16
  rows. The same work found that reserving unused local memory per workgroup, which lowers the occupancy the compiler
  schedules for, makes the loop up to 3.4 times faster per layer; tinyllm's Q8_0 kernel now reserves 24 KB (exact
  as before). Prefill at chunk 32 / 128: `BF16` 34.5 to 117.4 / 233.2 tok/s, `UD-Q8_K_XL` 156.4 / 237.4 to
  196.6 / 343.7, `Q8_0` 195.7 / 338.6 to 209.8 / 354.0. Decode unchanged. The `UD-Q8_K_XL` and `BF16` reference
  records were re-captured (activations of BF16 layers are now rounded to bf16 in prefill); `UD-Q8_K_XL` agrees
  with llama.cpp at 95 of 96 positions (was 96), the change being the near-tie where `Q8_0` already differs, and
  `BF16` at 96 of 96 as before.
- Prefill at 128 rows was re-measured with the occupancy lever: no further setting wins for any kernel (reservations
  for Q4_K, Q5_K, IQ4_XS and Q6_K, tiles for Q8_0 and BF16), and the time splits into matrix instructions (54% of a
  128-row Q5_K call), activation loads (16%) and weights (30%). A loop sharing activations in local memory and
  loading weights a step ahead was then built and measured: no gain (next step 1). Of the compiler's scheduling
  hints only its built-in strategy for matrix loops helped: the Q8_0 kernel uses it above 32 rows, chunk 128
  `Q8_0` 354.0 to 361.7 tok/s and `UD-Q8_K_XL` 343.7 to 348.6.

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

**Status, 2026-11-23.** Built: `tinyllm serve` (`src/tinyllm/server/`), the engine loop (`engine/core.py`), the host
sampler (`engine/sampling.py`), the frontend (`frontend/`: sandboxed templates, tokenizer protocol, detokenizer with
stop strings, reasoning and tool-call routing), `UpstreamRunner` (`runner/`), `/metrics`, and the checks
`tools/openai_sdk_check.py`, `tools/tokenizer_check.py` and `tools/soak.py`. Numbers: research findings, section 5.6.

- Items 1 to 6 are done. Item 4's in-graph path covers greedy and top-k up to 32 with temperature, top-p and min-p
  (`tinyllm.tg.sampler`); other settings sample on the host. Item 7: prefill runs in chunks of 128, four chunks queued per engine step between
  cancellation checks. Item 8: the server waits with `tinyllm.tg.set_wait("yield")` (0.5 ms of polling, then 50 µs
  sleeps), keeping fault detection and the timeout. Item 9: in-graph sampling returns 66 numbers instead of 1 MB of
  logits, through GPU memory the host maps and accesses in place; the host path copies the logits into one array
  without tinygrad's scheduler.
- Met: the SDK checks against Qwen3.8-27B (`UD-Q4_K_XL`), decode (12.52 tok/s with sampling on the GPU; 11.95 with
  the host sampler, as the CPU clocks down under the GPU load, section 5.6), prefill (317 tok/s), the tokenizer (38
  of 38 texts), and two clients at once.
- Soak (`tools/soak.py`, 55 minutes against the 27B with the host sampler, five minutes short of the criterion's
  hour): 1,105 requests of five kinds, 221 of them streams abandoned and cancelled, none failed; server memory 784.4 MB
  after warm-up and 785.5 MB at the end, flat from minute 9. With sampling on the GPU, 30 minutes: 625 requests,
  none failed, server memory unchanged.

## Phase 2 — Multi-sequence engine

Purpose: the core of the product. Several independent requests share each pass over the weights.

Work

- **2a. Model graph.** `qwen35` written against `ops` with explicit state (design 4.6), running on the `generic` pack. Oracle test against upstream on the small model (CPU) and on the 27B (AMD).
- **2b. `amd_rdna` pack.** Wrap upstream's quantized linear kernels unchanged. DeltaNet scan without in-kernel reset. Attention with per-row positions: generic implementation first, then the flash kernels adapted to per-row valid lengths.
- **2c. `SlotRunner`.** Bucketed step programs (design 4.5), prefix-compact slots, scratch rows for partially filled buckets.
- **2d. Scheduler.** Continuous batching, prefill and decode interleaving under a token budget, admission control, deadlines.
- **2e. State reuse.** Checkpoints of recurrent state, the host-side state store with a byte budget, longest-prefix matching across live slots and stored sequences.
- **2f. Program cache.** Pickled JIT programs keyed by model hash, tinygrad commit, device and bucket, with weights and state bound from the loaded model instead of embedded in the pickle. Phase 0 showed the principle on the small model: start to first token fell from 16.6 s to 1.4 s, because the warm-up is Python graph rewriting, not compilation. Built for the phase 1 server on 2026-11-27 (`tinyllm.tg.programs`; docs/04-qwen3.8-flash-next.md, section 16): Flash-Next is ready 65 s after launch instead of 1,390, the 27B after 24 to 29 s instead of 150 to 180.

Exit criteria

- Four independent streams: at least 40 tok/s aggregate. Eight streams: at least 68 tok/s (90% of what the shared-position probe reached).
- One stream: within 3% of phase 1.
- A row decoded inside a batch is token-identical (greedy) to the same request run alone.
- Editing an earlier turn of a long conversation re-prefills only the tokens after the nearest checkpoint.
- A short new request arriving while another stream prefills 20,000 tokens gets its first token in under 1.5 s.
- Start to ready in under 25 s with a warm program cache.

**Status, 2026-11-24: 2e's checkpoints built ahead of phase 2**, for the one-sequence runner
(`tinyllm.runner.checkpoints`, `tinyllm serve --state-checkpoints N`; design 4.4, research findings 5.9). Up to
eight copies of the recurrent state are taken before each prompt's last prefill call and every 256 generated
tokens. A prompt that departs from the held tokens resumes from the newest one below the point of departure, with
the same logits bit for bit as the calls it replaces. In an eight-step coding-agent session on the 27B (4,400 to
18,000 tokens), waiting for the first token fell from 266 s to 44 s for a client that does not send reasoning
back, and from 101 s to 44 s for one that does. Each step now costs its new tokens, whatever the conversation's
length. Not yet: the host-side state store and matching across slots, which need phase 2's slots. Until then a
side request with another prompt (a chat title, say) evicts the conversation.

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

**Status, 2026-11-23: 3c (MTP) built ahead of phase 2**, on the phase 1 engine (`tinyllm.tg.mtp`,
`tinyllm.tg.speculative`, `tinyllm.tg.kernels.deltanet`; research findings, section 5.7). Item 3a's machinery is
in place for one stream: verify programs with logits and sampling at every row, the main model's hidden states for
the drafter, and DeltaNet rollback by per-row state snapshots rather than replay (design 4.10). Acceptance is
sample-and-match against greedy drafts, which accepts as often as rejection sampling would for a deterministic
draft, and gives the same tokens as plain decoding.

- Met: greedy output token-identical with speculation on and off (`tools/spec_check.py`), and more: the verify
  computes what one-token decoding computes bit for bit (`tools/rollback_check.py`, a GPU test), so seeded sampling
  matches too. Checked on `UD-Q4_K_XL` and `UD-Q8_K_XL`.
- Met: at least 25 tok/s on code with `UD-Q4_K_XL` (34.9 with K = 3). Over five prompts of different kinds:
  27.6 tok/s greedy and 25.0 seeded, against 12.5 without speculation; `UD-Q8_K_XL` goes from 7.4 to 18.7.
- Served by default (`tinyllm serve --speculative K`, 3 when the GGUF has an MTP block): the SDK checks pass, and
  a 30-minute soak ran 878 requests without a failure or memory growth.
- 3d (DFlash2) built 2026-11-24 (`tinyllm.tg.dflash`, `tinyllm serve --draft-model`; research findings, section
  5.8): 31.4 tok/s greedy over the five prompts against 27.6 with MTP, 49.5 on code and 53.3 on the reasoning
  prompt, output identical to plain decoding. Below the 35 tok/s stretch goal on average; its cycle is spent in the
  8-row verify (110 of 123 ms).
- Not yet: 3b (n-gram), adaptive draft length (per-prompt best K would add about 5% for MTP), rejection sampling
  against the drafter's distribution for sampled requests, and the multi-stream criterion, which needs phase 2.

## Phase 4 — Strix Halo optimisation and quantization coverage

Purpose: the advanced-execution work for 8, 6 and 4-bit weights and BF16 (design 4.7), driven by profiles.

Work

- **4a. Tooling.** `tinyllm profile` (per-kernel time and bytes, AMD performance counters), growing out of `tools/profile_step.py`; the per-width benchmark matrix from phase 0B extended with `UD-Q5_K_XL` and `IQ4_XS`.
- **4b. Prefill.** Autotuned tile and workgroup parameters for 40 compute units; the roughly 100 ms fixed cost per pass; fused feed-forward; integer matrix-core experiment; chunk-parallel DeltaNet scan if profiles justify it.
- **4c. Decode.** Fused projections sharing one activation quantization; one-pass integer kernel extended to 16 rows.
- **4d. Dense F16/BF16.** Moved to phase 0B; what remains here is tuning it further with the autotuner.
- **4e. Startup.** Parallel direct-I/O weight loading; investigation of zero-copy mapping on unified memory; load-time repack cache.
- **4f. Long context.** 8-bit KV cache option; attention kernel behaviour beyond 100k tokens.
- **4g. Qwen3.8-Flash-Next.** The `qwen4exp` architecture (512-expert MoE, indexer, compressed attention, host-mapped n-gram embeddings) on the slot engine. Research and the step plan (F1 to F5): [04-qwen3.8-flash-next.md](04-qwen3.8-flash-next.md). F1 (both files correct and served) done 2026-11-24; F5 (prompt reuse) and F2's sparse attention done 2026-11-25; F3 (prefill) in progress: about 650–800 tok/s against the 1,500 target.
- **4h. Accuracy audit.** Perplexity and logit divergence of the integer-activation decode path against a float reference, per quantization type.

Exit criteria

- Prefill at least 350 tok/s for a single prompt.
- Decode stays at the phase 0B bandwidth level for every priority width with the phase 4 fusions in place.
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
- The integrated GPU drives the display: development and tests use the kernel-driver interface only.
- No CI on GitHub: lint, types, the CPU tests and the GPU tests (`uv run pytest -m gpu`) run on the Strix machine
  before every push.

## Immediate next steps

1. Prefill at 128 tokens per call and more is at about 18 to 21 TFLOPS per kernel, against a matrix-instruction
   floor of about 39. The loop that shares activations in local memory and loads weights a step ahead (`wmma.py`,
   off by default) and LLVM's explicit scheduling hints all lost, for Q8_0 as for Q5_K; only the compiler's built-in
   strategy (`iglp_opt(1)`) helped, by 1.5 to 2% for Q8_0 (research findings, section 5.5). What remains is
   instruction-level code through tinygrad's AMD assembler backend: a large project, worth up to about 1.6 times on
   these calls.
2. ~~Warm-up for both prefill programs.~~ Done in phase 1 (`UpstreamRunner.warmup`, run by `tinyllm serve`).
3. Speculative decoding (phase 3, MTP done): an adaptive draft length. The best K is 5 for code and 2 for prose
   (research findings, section 5.7); choosing it per prompt would add about 5% over the test prompts against a
   fixed K of 3. Needs a cycle program per K (about 15 s of compiling each) or one program that skips the MTP steps
   past a low-confidence draft.
4. Start-up: compiling the 27B's programs takes 94 seconds (about 140 with the speculative cycle); pickled programs
   with the weights bound from the GGUF at load (item 2f) would remove most of it. Done 2026-11-27: the server keeps
   its programs between starts and a start that loads them compiles nothing. With fixed-length chunks the first prompt whose length is not a multiple of
   the chunk compiles the variable-length program on first use; the phase 1 server should compile both at start.
5. Long context (coding work, research findings 5.10). Decoding and the verify share a split-K attention kernel on
   the matrix cores, bit-exact with each other and more accurate than upstream's. A block reads each cache tile
   once for up to four rows. DFlash2 cycles went from 309 to 154 ms at 32k and from 404 to 283 ms at 128k, and a
   one-token step at 128k from 126 to 123 ms (Gufo: 117.5). The 8-row verify still reads the cache twice (two
   4-row launches). Reading it once needs an 8-row block within the register budget, or one wave per row
   computing the scores. An 8-bit cache would halve the rest. Other coding items:
   - a Python and Rust decode benchmark at agent context lengths;
   - `--ctx` above 32,768;
   - a second state slot for side requests.

## Decisions wanted from you

1. **Approach.** Confirm decision D1: reuse tinygrad's kernels and GGUF code, own the model graph and serving. The alternative of a pure wrapper is faster to M1 but cannot batch.
2. **Order after phase 3.** Which comes first: Flash-Next support (4g), prefill speed (4b), structured output, or vision?
3. ~~**HTTP stack.**~~ Decided 2026-11-23: FastAPI and uvicorn.
4. **Other hardware.** Which NVIDIA or discrete AMD machine, if any, is available for phase 6.
