# Qwen3.8-Flash-Next: research and plan

Status: research 2026-11-24; step F1 (correct and served) done the same day, with its results and what it found
in section 11. Step F5 (prompt reuse) done 2026-11-25, section 12. F2's sparse attention done the same day,
section 13. F3 (prefill) in progress since the same day, section 14: about 650–800 tok/s, against the 1,500 target.
F4 (speculative decoding: the MTP block, DFlash2, copied blocks) done 2026-11-27, section 15. The rest of F2 is
not started. Start-up: the compiled programs are kept between starts since 2026-11-27, section 16.

Sources:
- the local GGUF files (unsloth `UD-IQ4_XS` and `UD-Q4_K_XL`);
- the official model card and `config.json` (`Qwen/Qwen3.8-Flash-Next`);
- llama.cpp b12675 (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`);
- the pinned tinygrad (`tinygrad/llm/model.py`, `gguf.py`, `kernels/`);
- Gufo's model notes and benchmarks (`docs/internal/models/qwen3.8-flash-next/`);
- Strata (`docs/STRIX_HALO.md`, `HOW_IT_WORKS.md`, `MODELS.md`);
- Radiance (`arch/qwen4exp_fp8`, `arch/common/rad_qsa.h`, `rad_block_ple.h`);
- a probe of upstream tinygrad on this machine (section 7).

## 1. Summary

Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`, "a preview of the Qwen4 architecture") has these parts:
- **Size:** 125B parameters with 6B active per token, plus a 51B hashed n-gram embedding table and a separate 4B MTP drafter.
- **Mixing layers:** the same Gated DeltaNet as the 27B, alternating with a block-sparse attention, QSA.
- **Experts:** a 512-expert mixture.
- **Residual:** four residual streams joined by low-rank gated mixers.
- **N-gram embedding:** one 28.8 GB table, read a few rows per token.

Two properties make it attractive on this machine:
- **Low bytes per token:** about 6 GB per decode token for `UD-Q4_K_XL`, against 17 for the 27B. The bandwidth floor is about 34 tok/s plain.
- **Near-flat attention cost:** attention reads at most 2,051 positions per token. Gufo decodes at 28.3 tok/s at depth 0 and 27.9 at 131k.

The pinned tinygrad already implements the architecture. On `UD-Q4_K_XL` it loads, fits in 76.9 GiB of GPU memory and writes correct text, but decodes at 6.9 tok/s, a quarter of Gufo's 28.3. It cannot load the current `UD-IQ4_XS`, which has Q2_0 tensors. The work is mostly kernels and session integration, not the model graph.

After F1 (section 11) both files run in tinyllm's session and server:
- **Decode:** 26.6 tok/s (`UD-Q4_K_XL`) and 29.3 (`UD-IQ4_XS`) with a 4,096-token context.
- **Agreement:** llama.cpp b12675 picks the same most likely token at 93–96% of positions.
- **What limits it now:** the sparse attention, which in tinygrad's generic path costs more as the context size and the position grow, and expert prefill.

After F5 (section 12) an agent's next step resumes from a checkpoint, as on the 27B. At an agent's positions, 4k to 17k tokens, decode fell to 5–12 tok/s and prefill to 35–60 tok/s: tinygrad's sparse attention read the whole history through generic code.

After F2's sparse attention (section 13) each query reads at most 2,051 positions through four kernels. In the agent benchmark decode holds 25–27 tok/s from 4k to 20k positions, and prefill runs at about 140 tok/s. The weights now take 77% of a decode step. What limits it next: expert prefill (F3), then the small kernels and host time of decode, then speculative decoding (F4).

After F4's MTP drafting (section 15) `UD-Q4_K_XL` decodes code at 52–55 tok/s and prose at 34–35, against 26–28 plain. The model's DFlash2 drafter runs too and is slower here. What is left of F4: output identical to plain decoding, and drafts copied from earlier text.

## 2. Architecture

Verified against `config.json`, the GGUF metadata and llama.cpp's implementation. Indices are 0-based.

| Part | What it is |
|---|---|
| Blocks | 48, width 2,560. Layout 12 × (3 × (Gated DeltaNet → MoE), 1 × (QSA → MoE)); QSA at blocks 3, 7, …, 47 |
| Residual (gated residual, "hyper-connections") | 4 streams of 2,560 (a 10,240-wide residual), initialised as 4 copies of the embedding. Before each mixer and each MoE: grouped RMSNorm over each stream, a rank-320 down projection (with 4 "inject" rows), SiLU, up projection, a sigmoid read gate, mean over the streams. After: each stream adds the branch output × 2·sigmoid(inject). The final mixer replaces the output norm (`output_hc_*`) |
| Gated DeltaNet | 16 QK heads and 48 V heads of 128, conv kernel 4, as on the 27B, but the output gate is a sigmoid (the 27B: SiLU) |
| QSA attention | 24 query heads, 2 KV heads, head size 256, output gate (sigmoid), q/k RMSNorm. Rotary on 64 dimensions (interleaved M-RoPE sections 11/11/10, θ 1e7). The cache holds the full K/V; the indexer picks what each query reads |
| QSA indexer | 4 query heads and 1 key head of 128 (`indexer.q_proj` 512 × 2,560, `k_proj` 128 × 2,560, BF16). Keys are pooled over blocks of 4 consecutive positions (mean of the raw keys → RMSNorm → RoPE at the block's first position). Score of a block = Σ over heads of relu(q_h · k) / √128. A token attends to the top 512 complete visible blocks plus its incomplete tail ((rank + 1) mod 4 positions, itself included): at most 2,051 keys. Up to 2,051 positions this is dense causal attention |
| MoE | 512 experts of width 640, 10 routed (softmax, normalised top-k) plus 1 shared expert with a sigmoid gate. Router weights are F32 (5.2 MB a layer) |
| N-gram embedding (PLE) | At block 1 (the model card's "layer 2"), before its mixer. 16 heads: 8 hash the bigram, 8 the trigram ending at the token, `(t0·m0) xor (t1·m1) [xor (t2·m2)]`, then `% vocab_h + offset_h` (vocabularies are primes of about 20M). An EOS in the window resets what lies before it, and a missing predecessor reads as EOS. Each head gathers one 160-value row of `per_layer_token_embd` (320,001,536 × 160, IQ4_NL, 28.8 GB). The 2,560-value result goes through a key projection (to 10,240) and a value projection (2,560), with grouped norms. The gate is sigmoid(sign(s)·√\|s\|) of each stream's key · query / √2,560. A depthwise causal convolution follows (kernel 4, dilation 3, over the 10,240 width, with its own state) |
| MTP | Separate GGUFs: one QSA block with its own experts, plus `nextn.*` (eh_proj over [enorm(e); hnorm(h)] per stream, head mixer). `shared` variants (`mtp-…-shared-Q8_0`, 2.8 GB; `Q4_K_M`, 1.9 GB) borrow the target's embedding and head (`nextn_shared_target_tensors`) |
| Vision | `mmproj-BF16.gguf` (27-block ViT, 1,152 wide, out 2,560); not in scope for the first steps |

## 3. The two files

| | `UD-Q4_K_XL` (4 shards) | `UD-IQ4_XS` (4 shards, the current upload) |
|---|---|---|
| Total | 111.3 GB | 93.7 GB |
| N-gram table | 28.8 GB IQ4_NL | 28.8 GB IQ4_NL (the same shard) |
| GPU part | 82.5 GB | 64.9 GB |
| Dense layers | Q8_0 | Q6_K (DeltaNet qkv/gate, attention q, head), Q8_0 (ssm_out, hyper-connections) |
| Expert gate/up | Q4_K (47 layers), Q5_K (1) | IQ3_XXS (28), IQ4_XS (17), IQ2_S (3) |
| Expert down | **Q5_1** (43 layers, 27.1 GB), Q8_0 (5) | **Q5_0** (33, 19.0 GB), IQ4_NL (13), **Q2_0** (2: blocks 8 and 10, 0.5 GB) |

The types in bold have no packed kernel in the pinned tinygrad (`kernels/amd_gguf.py`: Q8_0, Q2_K to Q6_K, the IQ types, MXFP4). Q2_0 is new: `d · (q − 1)` for 2-bit q, blocks of 64 with an F16 scale (18 bytes). Neither tinygrad's decoder nor llama.cpp before b12675 reads it, which is why this `UD-IQ4_XS` needs b12675. `tools/build_ref_llamacpp.sh` now builds that tag.

## 4. Memory on this machine

123 GB of RAM, a TTM limit of 122 GiB for the GPU, 2 GiB carved out as VRAM, NVMe storage.

- **GPU, `UD-Q4_K_XL`:** 76.9 GiB with upstream tinygrad, measured (section 7). The n-gram table stays in the mmap'd file. Gufo reports 88.8 GiB at a 133k context.
- **GPU, `UD-IQ4_XS`:** about 61 GiB.
- **Per context token:**
  - K/V: 24 KiB (12 layers × 2 heads × 256 × K and V × F16);
  - indexer keys: 3 to 6 KiB;
  - Gufo measured 27.5 KiB with its MTP drafter.
- **Recurrent state:** 36 DeltaNet layers × 3 MiB, plus the n-gram convolution state (9 × 10,240 floats).
- **Host, the n-gram table:** read through the page cache, 16 rows of 90 bytes per token.
  - With `UD-IQ4_XS` the table fits in RAM next to the weights (61 + 27 GiB).
  - With `UD-Q4_K_XL` it competes for the remaining 30 GiB, and cold rows come from the SSD. Strata ships it that way; Gufo measured cold rows costing 2 to 4% of decode in a first run.

## 5. Cost of one decode token

Bytes read, `UD-Q4_K_XL`:

| Part | Per token |
|---|---|
| DeltaNet projections, 36 layers (qkv 27.9 MB, gate 16.7, out 16.7, α/β F32 1.0) | 2.24 GB |
| QSA projections, 12 layers (q 33.4, k/v 2.8, out 16.7, indexer 3.3) | 0.67 GB |
| Gated residual mixers, 2 × 48 (4 × 3.5 MB each layer) | 0.67 GB |
| Routed experts, 10 × (Q4_K gate/up 1.84 MB + Q5_1 down 1.23 MB) × 48 | 1.47 GB |
| Shared expert and F32 router | 0.50 GB |
| Output head (Q8_0) | 0.68 GB |
| N-gram key/value projections | 0.03 GB |
| **Total** | **≈ 6.3 GB → 29 ms at 215 GB/s, about 34 tok/s** |

The dense layers, not the experts, are most of it. `UD-IQ4_XS` reads about 5.4 GB, about 40 tok/s at the same rate. Context adds little:
- attention reads at most 2,051 positions × 2 KB per layer;
- the indexer reads one 128-value key per 4 positions: 100 MB per token at 128k.

## 6. Reference engines on this machine

| Engine, file | Plain decode | Speculative decode | Prefill |
|---|---|---|---|
| Gufo, `UD-Q4_K_XL` (2026-10-18) | 28.3 tok/s at depth 0 → 27.9 at 131k | MTP: 36–40 mixed text; 117 repetitive (prompt lookup) | 1,801 tok/s (1,557 at 131k) |
| llama.cpp b11069, the same file (Gufo's table) | 22.2 → 8.0 at 131k | MTP: 31.7 mixed | 490 tok/s |
| Strata 0.1.40, `UD-IQ4_XS` | — | MTP and lookup, int8 KV: 53.8 at 8k, 51.4 at 128k | ~1,300 tok/s |
| Strata 0.1.40, `UD-Q4_K_XL` | — | 52.6 at 8k, 43.3 at 128k | ~1,170 tok/s |
| tinygrad (pinned, upstream code), `UD-Q4_K_XL` | 6.9 tok/s (4k context) | — | first token after 54 s, mostly compiling |
| tinyllm after F1, `UD-Q4_K_XL` | 26.6 tok/s (4k context), 17.8 in the server (32k context) | — | 82–108 tok/s |
| tinyllm after F1, `UD-IQ4_XS` | 31.6 tok/s (2k context) → 29.3 (4k) → 20.4 (32k) | — | — |

Radiance (checked 2026-11-25) is not in the table:
- **Different hardware:** its published Flash-Next numbers are on two Radeon AI PRO R9700 cards (gfx1201, about 5 times this machine's bandwidth together). One stream there takes 12.5 ms per decode step at about 2.4 accepted MTP tokens, 15.2 ms behind 178k of context. Prefill is about 6,700 tok/s on a 24k prompt. On one card, 4,096-token prefill steps run at 2,545 tok/s against 1,413 for 2,048-token steps.
- **Different files:** it reads only its own `.rad` containers, re-quantised from the BF16 checkpoint (113 to 175 GiB), not the GGUFs.
- **Strix Halo:** it has a gfx1151 kernel library (`libhalo`) with Flash-Next's kernels (`qsa_*`, `ple_*`, `ngram_ids`, `hc_*`, `moe_*`), but publishes no speeds on it.

Its `docs/QSA.md` is the most detailed description of the sparse attention, and the design reference for F2:
- **Skip the indexer below the bound.** While a query sees at most 2,051 positions, its selection is every block, so it skips the scoring and the top-k altogether. It decides by position, not by the configured context.
- **Cache each block key once,** when its fourth token arrives (64 bytes per token per QSA layer), plus a tail of up to 3 raw keys.
- **Score only the live history.** Scoring reads the block keys once per 4-row query tile. Both ops are narrowed to a host-side ceiling of the live context, never `max-model-len`.
- **Exact chunked top-k.** The selection splits the block list into chunks and takes a local top-k in each. It is exact and linear, with a fixed tie rule, so every form gives the same bytes.
- **Sparse attention is paged attention:** pages of 4 tokens and a 512-entry table.
- **In prefill, each query selects its own set.** Neighbouring queries share 53–73% of their pages, and tiles of 8 rows read the union of their pages once.
- **Serving settings:**
  - 2,048–4,096-token prefill steps;
  - recurrent-state checkpoints every 4,096 tokens;
  - an int8 KV cache.

Gufo's notes on how it gets there:
- custom HIP kernels for routed-expert prefill (Q5_1 down and Q4_K gate/up at 24–33 TFLOPS);
- one-launch decode mixers for the gated residual;
- 4,096-row prefill passes;
- host-side n-gram gathers with de-duplication;
- confidence-gated MTP chains;
- prompt lookup that copies up to 64 tokens per pass (code edits and tool output).

## 7. The pinned tinygrad today

`Transformer.from_gguf` handles `qwen4exp`:
- `GatedResidual` (low-rank hyper-connections);
- `Indexer(qsa=True)` with pooled keys;
- `PLELayer`: a `GGUFPackedTable` keeps the table's rows in the mmap'd file, `write_rows` hashes on the host and copies each call's rows into a staging buffer, and the device decodes them;
- `ExpertWeights` with `moe_<type>` kernels;
- DeltaNet with the sigmoid gate.

Probe (`UD-Q4_K_XL`, upstream code only, `generate`, greedy, 30-token prompt): loaded in 35.6 s, 76.9 GiB of GPU memory, first token after 54.4 s, then 6.89 tok/s. The text is a correct palindrome function.

llama.cpp b12675 on the CPU agrees:
- **The same file, the same prompt, greedy:** identical token ids for all 64 tokens (3 tok/s on the CPU).
- **The current `UD-IQ4_XS`:** b12675 loads it, Q2_0 included, and its first 16 tokens are the same.

So the model graph in the pinned tinygrad is right, at least where QSA equals dense attention.

One decode step, per-kernel times under `DEBUG=2`:

| Kernel | Time | Calls | Note |
|---|---|---|---|
| `r_5_40_2_16_4_20_2_2_2_2_2` | 34.3 ms (48%) | 43 | Q5_1 expert down projections on generic code: 0.8 ms a layer for 12 MB |
| `linear_q8_0` | 21.7 ms (31%) | 497 | the dense layers |
| `moe_q4_k` | 4.3 ms (6%) | 94 | expert gate/up, near bandwidth |
| the rest | 10.5 ms | 4,168 | |

That is 4,802 kernels and 70.8 ms of kernel time against 145 ms of wall time per token. The rest is host time: launches and the Python `generate` loop.

Two upstream limits beyond speed:
- **Attention reads the whole cache.** With an indexer mask, `TransformerBlock._attention` takes the generic attention path over the whole cache and masks the unselected positions. Decode cost then grows with context as for dense attention, and QSA's sparsity buys nothing.
- **The n-gram rows are written only by `generate`.** tinyllm's sessions call the compiled programs directly, so every call path (prefill, decode, sampled decode, verify) must write them itself.

## 8. What tinyllm needs

1. **Kernels for the missing expert types:** Q5_1 (`UD-Q4_K_XL`), Q5_0 and Q2_0 (`UD-IQ4_XS`). Expert down projections are 640 → 2,560, decode and prefill. Q2_0 also needs a decoder, since tinygrad has none.
2. **Session support:**
   - n-gram row writes before every call;
   - logits and the GPU sampler over the gated-residual head;
   - the n-gram convolution state and the indexer's pooled keys in state checkpoints (prompt reuse);
   - `tinyllm serve` picking the model up, including its chat template and tool calls.
3. **Sparse attention for decode:** gather each query's selected positions (at most 2,051) instead of masking the whole cache, and score the pooled keys with one kernel. This is what keeps decode flat with context. Done, for prefill too (section 13).
4. **Decode speed toward the 6.3 GB floor:**
   - the dense Q8_0/Q6_K projections at one row;
   - fused mixers (16 small projections a layer);
   - fewer kernels and less host time per token (4,802 now).
5. **Prefill:**
   - routed experts grouped by expert, on the matrix cores, for every expert type;
   - QSA prefill: dense up to 2,051 positions, block-sparse beyond;
   - n-gram gathers that de-duplicate rows and fetch cold pages in parallel. A 10k-token prompt needs 160k rows; read one at a time from the SSD that is about 13 s.
6. **Speculative decoding:**
   - the Flash-Next MTP drafter (`shared-Q8_0`): a gated-residual hidden state, its own QSA block and experts;
   - prompt lookup, the largest gain for code;
   - a bit-exact verify, which needs row-independent expert kernels and identical indexer selections per row.
7. **A reference oracle:** llama.cpp b12675 on the CPU for both files (6B active parameters run at usable speed there). Logit agreement goes in the GPU tests, as for the 27B. QSA's sparse selection must be checked beyond 2,051 tokens, where it departs from dense attention.

## 9. Plan

| Step | Work | Done when |
|---|---|---|
| F1. Correct and served (done, section 11) | Q5_1, Q5_0 and Q2_0 expert kernels (decode first, a correct prefill path); Q2_0 decoding; n-gram rows in every session call path; the model in `tinyllm serve` with its template; the llama.cpp b12675 oracle at short and long (> 2,051) contexts; goldens | Both files serve and agree with llama.cpp at the 27B's standard; `UD-Q4_K_XL` decodes at 20 tok/s or more |
| F2. Decode (the sparse attention done, section 13) | Experts at bandwidth; dense one-row projections; fused gated-residual mixers; the sparse decode attention; host overhead down | `UD-Q4_K_XL` 28 tok/s and `UD-IQ4_XS` 32 or more, within 5% from 4k to 128k |
| F3. Prefill (in progress, section 14) | Grouped expert prefill on the matrix cores in passes of 2,048 tokens or more (section 6), QSA prefill, n-gram gathers with prefetch | 1,500 tok/s or more on a 16k prompt (Gufo 1,801, Strata about 1,200–1,300) |
| F4. Speculative decoding (the MTP block and DFlash2 done, section 15) | Flash-Next MTP with a bit-exact verify; prompt lookup | Output identical to plain decoding (not met: equal up to rounding); 50 tok/s or more on code (met: 52–55; Strata 52.6 on `UD-Q4_K_XL`) |
| F5. Prompt reuse (done, section 12) | Checkpoints of the DeltaNet and n-gram state | The agent benchmark (`tools/agent_bench.py`) reuses as on the 27B |

Order:
- **`UD-Q4_K_XL` first:** upstream already runs it, so F1 is integration plus one kernel (Q5_1).
- **Then `UD-IQ4_XS`:** Q5_0 and Q2_0 kernels. It is the faster file, and the n-gram table fits in RAM next to it.

After F1, for agent-style coding (long prompts, many turns), the order is:
1. **Checkpoints with the n-gram convolution state (F5).** Done first, on 2026-11-25 (section 12): it was small, and without it every agent turn whose history the chat template re-renders prefilled the whole prompt again.
2. **The indexer's selection and the sparse attention (F2),** on Radiance's design (section 6). Done on 2026-11-25 (section 13): decode and prefill depend on neither the context size nor, past 2,051 positions, the position.
3. **Grouped expert prefill on the matrix cores (F3).** Expert layers are 82% of prefill time at short positions. Waiting for the first token is now the largest cost of an agent step (section 13).
4. **The rest of F2:** the small kernels and host time that remain between the weights' 31 ms and the step.
5. **Speculative decoding (F4).**

## 10. Open questions and risks

- **Memory with `UD-Q4_K_XL`.** 82.5 GB of weights plus caches leave about 30 GiB of RAM for the OS and the n-gram table's page cache. Cold rows come from the SSD. The first long prompt after start will pay for them unless gathers are parallel.
- **Bit-exact speculative verify.** On the 27B this depended on every kernel giving the same rows for one and several tokens. The expert kernels and the indexer's top-k (with its tie order) will need the same care.
- **QSA correctness beyond 2,051 tokens.** Checked since: the goldens include a 3,128-token prompt compared with llama.cpp (sections 11 and 13), and `tools/qsa_check.py` checks the kernels against float64 around the bound.
- **The `UD-IQ4_XS` upload changed recently.** Its 4-shard set with Q2_0 replaced a 3-shard set. Tests should record the shard hashes.

## 11. F1 results (2026-11-24)

### What was built

- **Expert kernels for Q5_1, Q5_0 and Q2_0** (`tinyllm.tg.kernels.moe_small`, kernel name `moe`):
  - Layout: tinygrad's decode-kernel layout (one wave per output and up to 1,024 inputs, one lane per group of 32 weights).
  - Arithmetic: each lane multiplies its 32 codes, as floats, by fp32 activations. A first version used tinygrad's int8 activations with `sudot4`. It ran at the same speed, so the version without activation rounding was kept: 1e-7 relative error against float64, where tinygrad's generic path, which rounds the weights to fp16, has 2–3e-7.
  - Speed: one token's down projection (10 experts of 2,560 × 640) takes 0.11 ms, against 0.85 ms on tinygrad's generic code. 32 tokens run at 211 GB/s (`tools/moe_kernel_check.py`).
  - Prefill goes through the same kernel, one row per token and expert, as tinygrad's own expert kernels do.
- **Q2_0 decoding:** `moe_small.install_decoders` adds type 42 to tinygrad's GGUF reader and decoder when the session module is imported. It matches ggml's `dequantize_row_q2_0` exactly (`tests/test_kernels.py`).
- **N-gram rows on every call path:** for this model only, the session keeps the ids it has fed and writes each call's rows before the call. This covers prefill chunks, decode with host-chosen tokens, decode with the GPU sampler, and sample mode, which uses the ids it read back. For now the model runs one row, without speculative decoding, and with state checkpoints off (F5).
- **N-gram rows read from the file** (`tinyllm.tg.ngram.FileRows`):
  - The problem: tinygrad copies rows out of its own mapping of the GGUF file, which `DiskDevice` sets up with `MADV_HUGEPAGE`. On this kernel and btrfs, a fault on a row not in the page cache then reads a large block of the file. A token's 16 rows of 90 bytes took 63–68 ms, twice a decode step.
  - The fix: `FileRows` reads from its own descriptor, opened with `POSIX_FADV_RANDOM`. A first pass of non-blocking reads (`preadv2` with `RWF_NOWAIT`, through ctypes) copies the cached rows in under a microsecond each. The rest go to 16 threads that read in parallel.
  - The result: 1.6 ms per token when the rows are cold (median under decode load), 0.03 ms when cached. On a file system without non-blocking reads (tmpfs) every row goes to the threads.

### Speed

Session decode (`UpstreamSession` driven directly), greedy, 30-token prompt:

| File | Context | Logits to the host | GPU sampler (as the server runs) |
|---|---|---|---|
| `UD-Q4_K_XL` | 4,096 | 21.5 tok/s | 26.6 tok/s |
| `UD-IQ4_XS` | 4,096 | 28.2 tok/s | 29.3 tok/s (27.6 with every n-gram row cold) |

One `UD-IQ4_XS` decode step at a 2,048 context runs 3,796 kernels with 34.0 ms summed under `DEBUG=2`:

| Kernel | Time | Share |
|---|---|---|
| `linear_q6` | 11.6 ms | 34% |
| `linear_q8_0` | 7.4 ms | 22% |
| `moe_iq3_xxs` | 2.1 ms | |
| `moe_q5_0` (33 layers) | 2.0 ms | |
| `moe_iq4_xs` | 1.4 ms | |
| generic code | | 15% |

**Decode follows the context size, not the position.** `UD-IQ4_XS`, GPU sampler, all at position 158:

| Context | Decode |
|---|---|
| 2,048 | 31.6 tok/s |
| 4,096 | 28.4 tok/s |
| 16,384 | 24.5 tok/s |
| 32,768 | 20.4 tok/s |

The cause is the indexer path:
- Above 2,048 positions of context, `_indexer_mask` runs the indexer at every call.
- Each QSA layer sorts max_context / 4 block scores with tinygrad's bitonic `sort`. Its index recovery compares all pairs (`mixin/op.py`), so it costs (max_context / 4)² per layer and token, at any position.
- Attention then takes the generic masked path over the cache instead of the decode kernel.
- At 131k this would be 16 times the 32k cost.

**Prefill:** one 128-token chunk of `UD-Q4_K_XL`, summed kernel time under `DEBUG=2`:
- **2,048 context:** 684 ms. 82% is the expert layers:
  - `moe_q4_k` 282 ms;
  - `moe_q5_1` 233 ms;
  - `moe_q8_0` 36 ms.

  They reread each expert's weights for every token routed to it.
- **32,768 context:** 1,098 ms. The extra 414 ms is the indexer's sort (three kernels of 233, 30 and 13 ms) and the masked attention (80 ms).

**Server:** `tinyllm serve` loads `UD-Q4_K_XL` in 32 s and then compiles for 266 s. `tools/openai_sdk_check.py`:
- **Pass:** streaming, reasoning on and off, stop sequences, seeds, a tool-call round trip and two clients at once.
- **Fail, the speed check:**
  - decode 17.8 tok/s at the default 32,768 context (the indexer's cost above; short requests showed 17.3–19.7);
  - prefill 82 tok/s on a 6,412-token prompt, 78 s to the first token.

### Correctness

Greedy records captured on the GPU (`tests/golden/qwen3.8-flash-next-*.amd.json`; the six standard prompts plus a 3,128-token source file, past the 2,051 positions where QSA becomes sparse) and compared with llama.cpp b12675 on the CPU (`tinyllm oracle reference`):

| Run | Same top token as llama.cpp | Mean / max log-probability difference |
|---|---|---|
| `UD-Q4_K_XL`, tinyllm | 104 / 112 (92.9%) | 0.060 / 0.93 |
| `UD-IQ4_XS`, tinyllm | 108 / 112 (96.4%) | 0.044 / 0.44 |
| `UD-IQ4_XS`, tinygrad's own code (`TINYLLM_KERNELS=none`) | 107 / 112 (95.5%) | 0.035 / 0.55 |
| `UD-IQ4_XS`, tinyllm without `moe` | 111 / 112 (99.1%) | 0.034 / 0.83 |
| For comparison, Qwen3.8-27B `UD-Q4_K_XL` (doc 01, section 5.5) | 95 / 96 | 0.012 / 0.15 |

In every run our token is in llama.cpp's top 8, and the tokenizers agree.

The spread between runs is mostly the method:
- Each record follows its own greedy path, so after one near-tie flip a prompt continues on different text and collects further disagreements. Four of the eight `UD-Q4_K_XL` disagreements and three of the four `UD-IQ4_XS` ones are in a single prompt.
- The router picks 10 of 512 experts, and small differences change that pick; the dense 27B has no such step.
- In isolation the `moe` kernels are closer to exact arithmetic than the generic path they replace.

Greedy decoding of the palindrome prompt gives llama.cpp's exact tokens on both files: 64 of 64 on `UD-Q4_K_XL`, 16 of 16 on `UD-IQ4_XS`.

### Against F1's done-when

- **Both files serve, tools work:** yes.
- **Agreement with llama.cpp:** top-1 agreement 93–96% against the 27B's 99%, never outside llama.cpp's top 8; log-probability differences four to five times the 27B's, for the reasons above.
- **`UD-Q4_K_XL` at 20 tok/s or more:** met in the session at a 4,096 context (26.6), not in the server at its default 32,768 (17.8). The fix is F2's first item, the indexer's selection.

## 12. F5 results (2026-11-25)

### What was built

- **State in the checkpoints.** A checkpoint now copies the n-gram embedding's convolution state (9 × 10,240 floats, 360 KiB) as well as the 36 DeltaNet blocks' recurrent and convolution states: 113 MiB in all (`UpstreamSession._state_tensors`). The other state is indexed by position and needs no copy: the attention cache, the indexer's raw keys (tinygrad pools and normalises them again on every call) and the staged n-gram rows.
- **N-gram history on resume.** `load_state` cuts the n-gram history back to the checkpoint. The next rows hash the two ids before them, and the runner only resumes from checkpoints that lie on the tokens it holds, so those ids are the checkpoint's.
- **Tests.**
  - The CPU suite checks which state is copied and the history cut, on stand-in blocks (`tests/test_checkpoints.py`).
  - The GPU suite runs `tools/checkpoint_check.py` on `UD-IQ4_XS` with 1,024-token chunks (`tests/test_kernels.py`).

### Correctness

`tools/checkpoint_check.py` on `UD-IQ4_XS`, greedy, sampling on the GPU. Each resumed request must give, bit for bit, the logits of running the same calls without the detour:

| Chunk | Resumed at | Logits | Tokens |
|---|---|---|---|
| 128 | 256 (before the prompt's last chunk) and 369 (while decoding) | identical | identical |
| 1,024 | 2,048, and 2,609 (past the bound, where the indexer selects) | identical | identical |

### The agent benchmark

`tools/agent_bench.py` against `tinyllm serve` on `UD-Q4_K_XL`:
- **Server:** 32,768 context, 8 checkpoints, 128-token chunks.
- **Session:** eight steps, greedy with thinking on, 400 tokens per reply, the client not sending reasoning back.

| Step | Prompt | Reused | Prefilled | First token after | Decode |
|---|---|---|---|---|---|
| 0 | 4,402 | 0 | 4,402 | 51.2 s | 11.6 tok/s |
| 1 | 7,220 | 4,352 | 2,868 | 46.1 s | 8.6 |
| 2 | 8,727 | 7,168 | 1,559 | 29.5 s | 8.0 |
| 3 | 10,017 | 8,704 | 1,313 | 27.0 s | 7.3 |
| 4 | 11,533 | 9,984 | 1,549 | 34.7 s | 6.7 |
| 5 | 13,074 | 11,520 | 1,554 | 38.4 s | 6.2 |
| 6 | 14,405 | 13,056 | 1,349 | 35.0 s | 5.8 |
| 7 | 17,445 | 14,336 | 3,109 | 89.4 s | 5.1 |

**Reuse works as on the 27B.** Each step prefills the tool output it adds plus at most one chunk: 13,301 tokens over steps 1 to 7, against 11,400 for the 27B's run (doc 01, section 5.9). Before F5 the server took no checkpoints for this model, so these steps would have prefilled all 82,421 prompt tokens. That is an estimated 17 to 30 minutes at this file's prefill rates, against 300 s.

**Speed at these positions is far below the 27B's.**
- **Waiting for the first token:** 27 to 89 s per step, against 5 to 6.5 s for the 27B.
- **Prefill:** the increments ran at 35 to 62 tok/s, 44 on average.
- **Decode:** it fell with the position, from 11.6 tok/s at 4.4k to 5.1 at 17k, 6.8 on average (2,522 tokens in 370 s, from the server's `/metrics`). The 27B ran at 20 to 34 tok/s with DFlash2 in the same benchmark.

F1 measured decode only at position 158, where QSA is still dense. Past 2,051 positions tinygrad's QSA path builds a mask over the whole history and runs generic attention over it, and it pools, normalises and ropes every block key again on each call (section 7). The decode step grew by about 110 ms between 4.4k and 17k positions. That is far more than the 0.3 GB of extra attention cache it reads, so the generic kernels themselves are slow.

A profile of one decode step at position 16,000 (`tools/profile_step.py --kind sampled --context 16000 --max-context 32768`, `UD-Q4_K_XL`) shows 5,305 kernels and 196 ms in all:

| Part | Time | Note |
|---|---|---|
| Attention in the 12 QSA layers (`r_3_4_8_16_4_(p4+1)`, `r_24_(p5+1)_16_16`, `r_2_3_4_(p3+1)`, `r_2_3_4_(p2+1)`, `E_(p3+1)`) | 138 ms (70%) | tinygrad's generic masked attention: with an indexer mask it leaves the flash-attention kernel (`TransformerBlock._attention`) and reads all 16,000 positions, about 11.5 ms a layer, where 2,051 positions of cache take about 0.02 ms to read at this machine's bandwidth |
| The indexer's scoring and selection | about 15 ms | follows the context size (the 32k session; section 11) |
| Dense Q8_0 layers (`linear_q8_0`, 497 calls) | 22.9 ms | 4.8 GB: at the bandwidth floor |
| Experts (`moe_q4_k`, `moe_q5_1`, `moe_q8_0`) | 7.9 ms | 1.5 GB: at the bandwidth floor |
| The rest (mixers, norms, router, n-gram, DeltaNet) | about 12 ms | thousands of small kernels |

So the weights, which a 6B-active model keeps small, cost what they should: about 31 ms, against about 80 ms for the 27B's 17 GB. The step is slow because attention runs through the generic path over the whole history: it is 70% of the step at 16k and grows with every token. With F2's sparse attention, attention and selection cost about 1 ms at any position, and the step would be about 45 ms (about 22 tok/s) at 16k as at 158. This is the first thing F2 must fix: the attention reads at most 2,051 positions whatever the position.

### Against F5's done-when

- **The agent benchmark reuses as on the 27B:** yes. The cached tokens follow the conversation and each step prefills only what it adds.
- **Not measured:** a run with `--state-checkpoints 0` for a measured before (estimated above), and the `UD-IQ4_XS` file in the benchmark.

### Other requests between the turns (2026-11-26)

A chat in Jan showed that reuse failed whenever a client sent another request between the turns. Jan asks for a chat title after every reply: a 454-token prompt that shares nothing with the conversation. The server log of a conversation that grew to 23,221 tokens:

| Turn's prompt | Reused | First token after |
|---|---|---|
| 1,336 | 0 | 8.4 s (6.3 s of it queued behind the title request) |
| 8,578 | 0 | 15.4 s |
| 14,570 | 0 | 20.7 s |
| 23,221 | 0 | 31.1 s |

The checkpoints only described the sequence the state held. The title request replaced that sequence, so every checkpoint was dropped, and the next turn prefilled the whole conversation.

What a side request really destroys is small: the recurrent state, which a checkpoint puts back, and the cache entries of its own positions (0 to about 580 here). The conversation's checkpoints and its cache entries past those positions are intact. So (`tinyllm.runner.checkpoints`):

- **A checkpoint remembers the tokens it stands for** and matches any prompt that starts with them, whatever the state holds now.
- **The runner records which token sequence wrote each cache position** (`Ownership`). A position serves a prompt only if that sequence agrees with the prompt up to it.
- **`plan` picks the cheapest way back.** It runs the positions another request overwrote again, then loads the conversation's deepest checkpoint, whose cache entries in between are valid, and runs the rest. For the 23,221-token turn above that is about 580 + 1,000 tokens instead of 23,221.
- **`feed` makes the jump.** The engine still hands the prompt over in slices from the position `begin` returns; a slice the jump already covers runs nothing. The log's `cached` is the number of tokens not run.
- **Eviction** goes by request: the oldest request's checkpoints first, a decoding checkpoint before a prompt one, a lower position before a higher. A title request cannot push out the conversation's last prompt checkpoint.

Cache entries a speculative verify writes past the accepted tokens belong to no one.

**Checks.**
- `tests/test_checkpoints.py` (CPU, the development model): a request of one chunk between two turns, with the prompt fed in one, two or three slices. The logits equal the uninterrupted conversation's bit for bit, because the positions run again are one call identical to the first one.
- `tools/checkpoint_check.py`, request 4, on the GPU: bit-identical on the development model and on `UD-IQ4_XS` with 1,024-token chunks (a 1,024-token request between the turns; 1,280 tokens reused).

**A chat with a side request after every reply**, `UD-Q4_K_XL`, 2,048-token chunks (a scratch script driving the runner as the engine does; the side request is 454 prompt tokens and 128 generated ones; replies are not sent back):

| Turn's prompt | Reused | Run | First token after |
|---|---|---|---|
| 1,200 | 0 | 1,200 | 2.3 s |
| 1,700 | 442 | 1,258 | 3.2 s |
| 8,000 | 954 | 7,046 | 8.4 s |
| 8,600 | 7,354 | 1,246 | 2.9 s |
| 14,500 | 7,866 | 6,634 | 8.3 s |
| 23,000 | 13,754 | 9,246 | 11.1 s |
| 23,700 | 22,202 | 1,498 | 3.4 s |
| 23,700, no side request before it | 22,784 | 916 | 2.0 s |

Each turn runs the 582 positions the side request overwrote plus what the conversation added since its last checkpoint. The last two rows are the same turn with and without the side request: their logits differ by at most 1.47 (mean 0.16), with the same top token. That is the rounding noise of running those 582 positions in a call of another length (section 14 measures 0.8 to 1.1 for a change of call length at 2,600 tokens).

What is left in those waits is the speed of short calls: the 1,498 tokens ran at 437 tok/s, against about 1,000 for full 2,048-token calls.

## 13. F2, the sparse attention (2026-11-25)

### What was built

`tinyllm.tg.kernels.qsa`, kernel name `qsa` (on by default). It takes over a QSA layer's attention (`TransformerBlock._attention` and `_indexer_mask`, and `Indexer._init_state` for one new cache) whenever the indexer selects, that is when the context exceeds 2,048, on gfx11. Below that tinygrad's own dense flash attention runs, as before. Four kernels, on Radiance's design (section 6):
- **Block keys.** Each call recomputes the keys of the blocks its tokens fall in (the mean of 4 raw keys, RMSNorm, rotary at the block's start) into a cache of block keys: 128 floats per block, 4 MiB per layer at a 32,768 context. A block's key is final when its fourth token arrives. The cache is indexed by position, so it needs no checkpoint.
- **Scores.** One thread per query and complete block of the live history, not of the configured context.
- **Selection.** One workgroup per query, exact and linear: a 16-way search over the scores' float bits finds the 512th largest score in 8 passes (scores are sums of relus, never negative, so their bits order like their values). Then each thread counts its run of blocks above it and equal to it, a prefix sum places the runs, and the list is written in ascending order. Equal scores at the threshold are taken lowest block first. A query that sees at most 512 blocks takes them all without reading a score. The list ends with the query's incomplete block.
- **Attention.** Paged attention over the list on the matrix cores: 8 list entries (32 keys) per tile, `decode_mma`'s tile loop with one query row per workgroup and the row a grid axis, so each prefill row gathers its own keys. A second kernel combines the splits (32 for decode, 4 for prefill).

Tests:
- `tools/qsa_check.py` in the GPU suite;
- the hook on the CPU (`tests/test_kernels.py`);
- both Flash-Next goldens re-captured on the new path.

### Correctness

**The kernels, against float64** (`tools/qsa_check.py`). Random data at Flash-Next's shapes, decode rows at positions 0 to 8,190 around the bound, and 100- and 128-row chunks across it, padded and symbolic:
- block keys within 1.3e-7 of the largest value;
- scores within 5e-7;
- every list exactly the top 512 of the kernel's own scores, ascending, then the tail block;
- attention within 4.7e-4, the precision of half-precision probabilities, as in decoding's kernel.

**The model, against llama.cpp b12675** (the golden prompts, the 3,128-token source file included):

| Run | Same top token as llama.cpp | Mean / max log-probability difference |
|---|---|---|
| `UD-Q4_K_XL`, F2 | 105 / 112 (93.8%) | 0.065 / 0.86 |
| `UD-Q4_K_XL`, F1 (section 11) | 104 / 112 (92.9%) | 0.060 / 0.93 |
| `UD-IQ4_XS`, F2 | 108 / 112 (96.4%) | 0.047 / 0.65 |
| `UD-IQ4_XS`, F1 | 108 / 112 (96.4%) | 0.044 / 0.44 |

The agreement is unchanged. One `UD-IQ4_XS` disagreement moved into the long prompt: at step 3 our two best tokens are 0.07 apart, where F1 had them 0.29 apart in llama.cpp's order.

**Against tinygrad's path,** the logits move by 0.2 to 0.6 even on short prompts, where QSA is dense attention. That comes from precision, not selection:
- tinygrad's QSA path computes attention in float32;
- the matrix-core kernels, ours and tinygrad's own flash attention, use half-precision queries and probabilities;
- tinygrad's flash attention, run at a 2,048 context on the same short prompts, moves the logits just as much (0.25 to 0.57, the mean of each step's largest difference).

With `qsa` disabled, the old records are reproduced exactly.

**Prompt reuse:** `tools/checkpoint_check.py` with 1,024-token chunks still resumes bit for bit, at 2,048 and 2,609.

### Speed

One decode step at position 16,000, the same profile as section 12:

| | F5 | F2 |
|---|---|---|
| Kernels | 5,305 | 3,884 |
| Kernel time | 196.0 ms | 40.5 ms |
| Attention, scoring and selection in the 12 QSA layers | about 153 ms | 1.1 ms (`qsa_select` 0.46, `qsa_attention` 0.36, `qsa_scores` 0.15, `qsa_attention_combine` 0.14) |
| Weights (`linear_q8_0`, `moe_*`) | 31 ms | 31 ms, now 77% of the step |

In `tools/qsa_check.py --context 32768 --time 16000` one layer takes 0.14 ms to decode and 1.85 ms for a 128-row prefill chunk. The server also compiles faster: 156 s against 275 s.

**At position 100,000** (a 131,072 context) the same profile gives 38.7 ms of kernel time: the step does not grow with the position. The part that does grow is the sparse attention's scoring and selection, which read every block's score: 4.5 ms of that step with the binary search as first written (`qsa_select` 3.1 ms, `qsa_scores` 1.0). A 16-way search, with 8 passes over the scores instead of 31, gives the same lists (the golden record is reproduced exactly). In `tools/qsa_check.py` at 100,000 positions one decode layer then takes 0.27 ms instead of 0.33: about 3.3 ms a step, against 1.7 ms at 16,000.

The agent benchmark, run as in section 12:

| Step | Prompt | Reused | Prefilled | First token after | Decode | Decode with F5 |
|---|---|---|---|---|---|---|
| 0 | 4,402 | 0 | 4,402 | 28.9 s | 27.3 tok/s | 11.6 tok/s |
| 1 | 7,220 | 4,352 | 2,868 | 19.4 s | 27.0 | 8.6 |
| 2 | 8,727 | 7,168 | 1,559 | 11.0 s | 25.8 | 8.0 |
| 3 | 10,017 | 8,704 | 1,313 | 9.4 s | 26.1 | 7.3 |
| 4 | 11,533 | 9,984 | 1,549 | 11.2 s | 25.6 | 6.7 |
| 5 | 12,832 | 11,520 | 1,312 | 9.5 s | 26.7 | 6.2 |
| 6 | 18,460 | 12,800 | 5,660 | 39.8 s | 25.1 | 5.8 |
| 7 | 19,968 | 18,432 | 1,536 | 10.7 s | 25.6 | 5.1 |

From step 5 the conversation differs from section 12's (the model read four files at once), as any change of arithmetic can make it.
- **Decode** no longer falls with the position: 25.1 to 27.0 tok/s on the 400-token replies from 7k to 20k, against 11.6 falling to 5.1.
- **Prefill** of the increments runs at about 140 tok/s, against 35 to 62.
- **Waiting for the first token** after the first step: 111 s in all, against 300 s, for 20,199 prefilled tokens against 17,703.

### Against F2's done-when

- **28 tok/s on `UD-Q4_K_XL`:** not yet, 25 to 27. The weights take 31 of the step's 40.5 ms of kernel time. The rest is thousands of small kernels (mixers, norms, router, n-gram, DeltaNet) plus host time, the remaining F2 items.
- **Within 5% from 4k to 128k:** decode is flat within the noise from 7k to 20k in the benchmark. The step's kernel time is 40.5 ms at 16k and 38.7 ms at 100k (the expert reads vary from token to token). The sparse attention adds about 1.6 ms between those positions, 4% of a step. Decode at 100k was not measured in the server.
- **`UD-IQ4_XS`:** not measured in the benchmark.

## 14. F3, prefill (in progress, 2026-11-25)

Target: 1,500 tok/s or more on long prompts, as Gufo (1,801) and Strata (about 1,200–1,300) reach on this machine.
Starting point (F2): about 140 tok/s in the agent benchmark, 128-token chunks.

### What was built

- **Grouped expert kernels** (`tinyllm.tg.kernels.moe_mm`, kernel name `moe_mm`): prefill calls of 32 tokens or more.
  - A route kernel sorts the (token, expert) pairs by expert.
  - Each expert's rows are cut into tiles of 64 rows plus one tile of 16, 32, 48 or 64 rows for the rest.
  - Each tile runs on the matrix cores with the expert's weights dequantized once for all its rows.
  - Gate and up are one kernel; the SwiGLU goes in its stores, in fp16, as the down projection's input.
  - Types: Q4_K, Q5_K, Q5_1, Q8_0 (all of `UD-Q4_K_XL`'s experts) and IQ3_XXS, IQ4_XS, IQ2_S, Q5_0, IQ4_NL, Q2_0 (all of `UD-IQ4_XS`'s). The IQ types decode through tinygrad's codebook tables and finish with one packed multiply per pair of weights. IQ4_XS loads its scales and codes as 8- and 16-byte vectors.
  - A workgroup past the routing's tiles runs no step. tinygrad's cost estimates cannot evaluate a loop count read from memory, so these kernels carry their own estimates.
  - A block's gate/up and down share one routing, and the route kernel reads 16 pairs per load: 57 → 24 ms per pass.
  - **Since 2026-11-26, one block kernel instead** (`_block_kernel`, kernel names `moe_block_*`). Each expert's rows are cut into tiles of up to 128 rows. A workgroup of 4 waves takes one tile and 64 outputs. Each wave computes 16 outputs of one weight matrix for all the tile's rows, 8 matrix tiles of 16 rows:
    - **The weights are the matrix instruction's A operand.** The instruction reads an even row from the low half-wave's lane and an odd row from the high half's (Gufo's finding; the kernel's results confirm it on this chip). So each lane decodes 8 of a fragment's 16 inputs and takes the other 8 from its partner lane with one `v_permlanex16` per word. Each weight is decoded once per tile.
    - **Weights load a block at a time.** Each lane loads its share of 256 inputs of its weight row in one go (4 groups of 32 for the 32-weight types when the inputs are not whole blocks of 256), a block ahead, and uses it over the block's steps. The step loop is unrolled over the block.
    - **Activations are staged in local memory** 32 inputs at a time. The local-memory writes, reads and barriers are C blocks whose barriers fence local memory only: tinygrad's own barrier waits for every global load in flight, which would wait for the prefetched weights.
    - **No instruction for matrix tiles a tile's rows don't reach.** A 16-row tile issues one matrix instruction per step where a full tile issues eight, so small experts take a 128-row kernel too: a matrix tile past the tile's rows issues nothing.
    - **Gate/up is two launches,** gate into an fp32 scratch buffer, then up with SwiGLU in its stores. A gate/up pair in one wave needs 128 accumulator registers and cannot also hold a block of weights.
- **The residual mixer's norm and gates** (`tinyllm.tg.kernels.hc`, `hc`): one kernel per token reads the 4 × 2,560 streams once and writes the fp16 normalised streams (the down projection's input), each stream's 1 / rms and the inject gates. The mix rebuilds the fp32 normalised streams from those, as tinygrad's computes them.
- **A tiled matrix-core kernel for Q8_0 and Q6_K** (`tinyllm.tg.kernels.gemm`): calls of whole 128-row blocks.
  - A workgroup of 8 waves computes 256 rows × 128 outputs, each wave a 64 × 64 tile.
  - Each step stages 32 inputs of activations and dequantized weights in local memory, shared by all waves.
  - Split-K for narrow layers.
  - Per layer at 2,048 rows: 29–30 TFLOPS against about 20 for the per-wave loop, on the 27B's shapes (`tools/prefill_kernel_check.py`).
  - It also takes the 27B's 128-token calls, where it is neutral (`tinyllm bench --chunk 128 --prefill 2048`, tiled against per-wave). Q8_0: 362.9 against 364.9 tok/s. `UD-Q4_K_XL`: 358.1 against 361.8. At 128 rows it needs split-K, and summing the splits costs 46 of the pass's 342 ms of kernel time.
- **Grouped sparse attention** (`qsa`, prefill calls of 64 rows or more). Groups of 4 consecutive rows share their keys. A union kernel merges the group's 4 lists and records, per block, which rows listed it. The attention kernel runs the group's 48 queries (4 rows × 12 query heads per KV head, three matrix tiles, no padding) over the union, gathering each 32-key tile once and masking each key by its row's mark and position. It normalises in place, so no combine kernel runs. The one-row kernel had all 4 waves compute the same scores: 160 matrix operations per row and tile where 64 are needed.
  - The union for any context (2026-11-28). As first written it kept one mark per 4-position block of the whole configured cache in local memory, capped at 60 KiB, so only contexts up to about 61,000 tokens took the grouped kernel; larger ones fell back to the one-row kernel, and section 13's claim that prefill does not depend on the context size no longer held. The lists are ascending and name each block once, so the union is now a merge of 4 sorted lists of at most 513 entries: a block belongs to the first list that holds it, and an entry's place is the number of distinct blocks below it, counted per list by a binary search and a prefix sum of owned entries. Its local memory (32 KiB) is set by the lists' length, and its work no longer grows with the position. No store is gated: tinygrad simplifies a gated store under its gate, and with the prefix sums in the addresses it copied the whole scan once per gate (positions past a list's end hold a sentinel instead, and write the union's length). `tools/qsa_check.py` checks the unions against numpy, `--context 262144` at the model's full context. 16,384 tokens of code (`UD-Q4_K_XL`, 2,048-token calls):
    - At a 32,768 context the old and new unions give bit-identical logits, at 1,270 tok/s both.
    - At 262,144 the old code's one-row fallback ran 1,156 tok/s and the merge runs 1,261.
    - With BEAM off, every call's logits at 262,144 are bit-identical to those at 32,768. With `JITBEAM=2` the last logits differ by up to 0.53 (same top token). The likely cause, not checked kernel by kernel: BEAM's choices are cached by kernel shape, so the kernels whose shapes hold the cache length are searched again and may sum in another order.
    - Prefill still slows with position, a cause not yet profiled: the 16,384-token calls climb from about 1,500 to 1,660 ms, and the server ran a 166,705-token prompt at 943 tok/s.
    - `tinyllm serve` now defaults this model to the context its file states, 262,144 (`serve.ARCH_NATIVE_CTX`; other models keep 32,768). The full cache takes about 7 GiB more than 32,768.
  - Each query tile has a pair of waves that split the 256 output dimensions. Since 2026-11-26 they also split each key tile's scores, 16 keys each, instead of both computing all 32. They exchange their rows' maxima through local memory (one more barrier per tile) and share one probability tile. Their partial row sums travel with it, at the barrier before the value product. That is a third fewer matrix operations. 16,384 tokens of code: 861 → 878 tok/s, the later calls about 60 ms faster.
- **The residual mix in the up projection's stores** (`gemm.mix_linear`). The tiled gemm gives each wave's four 16-output tiles the same 16 dimensions of the four streams. Its stores then form mean(sigmoid(up) × stream / rms × weight), and the 2,048 × 10,240 fp32 up-projection output is never written. Logits are bit-identical. 16,384 tokens of code: 780 → 795 tok/s.
- **fp16 expert outputs** (`moe_mm.HALF_DOWN`). The down projection writes its 2,048 × 10 × 2,560 outputs in fp16 instead of fp32, and the weighted sum over each token's experts reads them back in fp32. Calls are about 100 ms faster (16,384 tokens of code: 1,822–2,053 against 1,921–2,151 ms). On 2,600 tokens of code the last logits move by at most 0.80 (mean 0.11). A control that only changes the call size moves them by 0.93 (mean 0.13).
- **The n-gram rows off the critical path** (`tinyllm.tg.ngram`, `UpstreamSession._prefetch_ple`). Before each prefill call the host computes and copies 16 table rows per token. That took up to 3.7 s of a 16,384-token prompt. Three changes:
  - **Written ahead.** A background thread writes the next call's rows while the current call runs. The host table has a slot per position, so the running call's rows are not touched.
  - **Gathered from a mapping.** numpy gathers the rows from a mapping of the file (MADV_RANDOM, no huge pages), split over a thread pool so that page faults read from the NVMe in parallel. The old reader made one `preadv2` call per row from Python.
  - **Hashed in numpy** (`row_ids`). tinygrad's `write_rows` builds Python lists per position. Inside a model process the collector then walks millions of objects: 999 ms for 2,048 positions, against 4.7 ms in an empty process. The numpy version takes 0.2 ms and gives the same rows, checked against `write_rows`.

  Time outside the calls on 16,384 tokens of code went from 3.0–3.3 s to 0.02 s, with bit-identical logits. With the server's wait (`yield`, which releases the interpreter lock) the prompt runs at 1,016 tok/s.
- **The router** (F32 in the file, rounded to fp16 by tinygrad's loader) takes the matrix cores in prefill (`dense_mm`, F32 words): 0.94 ms at 2,048 rows against 4.2 ms on tinygrad's one-row kernel. Since 2026-11-28 its large calls run on the tiled kernel ("The expert router on kernels of its own", below).
- **Prefill passes of 2,048 tokens** (`tinyllm serve` default for this model).
  - A third program for calls of at most 256 tokens.
  - The prompt's last call starts at its last multiple of 256, so the checkpoint before it lies close to the prompt's end. The checkpoint is also on a grid every request shares: a request that resumes there makes the same calls as the one that took it. A first version that started the last call 256 tokens before the end made a replayed request diverge (`tools/checkpoint_check.py`).
  - In a symbolic chunk, the matrix-core kernels skip the blocks of padding rows (bit-identical logits with and without).
  - The sparse attention skips its padding rows.
- **BEAM search for tinygrad's generic kernels** (`--beam`, default 2 for this model). It runs at the first start (573 s against 156 s to compile) and is cached on disk after.
- **The DeltaNet block on kernels of its own** (2026-11-27; `tinyllm.tg.kernels.deltafuse` and `deltanet.gated_delta_rows`). Fixed-length calls of 32 rows or more, on any model with this block (the 27B too).
  - **Around the scan, three kernels** (`delta_qk`, `delta_v`, `delta_out`). tinygrad builds the convolution's window, sums its taps, applies SiLU, splits the result into queries, keys and values, normalises queries and keys, repeats them for the value heads that share them, transposes all three, and after the scan normalises and gates the outputs. That is about ten generic kernels, each a pass over the call's data: about 147 ms a call, as much as the scan. The three kernels read the projection's output once and write what the scan reads: 54 ms (output norm and gate 23, values 18, queries and keys 13).
  - **Queries and keys once per key head.** The scan reads head h modulo the key heads, where tinygrad wrote a copy per value head.
  - **The scan with a state row on four lanes** (`gated_delta_span4`). tinygrad's layout spreads a row over all 32 lanes, four values a lane. Each of a row's two dot products is then summed across the wave, five swizzles and five adds: 80 of the about 170 vector operations per token for a wave of four rows. With a row on four lanes, a lane sums 32 products itself and the row's sum takes two swizzles, and a wave holds eight rows. That is about 21 operations per row and token against 42: 150 → 101 ms a call.
  - **Padded calls** (section 15): the scan runs as two kernels, the first leaving the state at the kept row in its buffer; rows past the prompt's end have beta 0 and decay 1.
  - The lane index of that scan is tinygrad's `WARP` axis, the wave's own lane. A `LOCAL` range that a kernel uses through `// 4` and `% 4` is split by tinygrad into two hardware dimensions (8 by 4 here), and sums over a row's four lanes then mixed four rows. tinygrad's own scan uses its range whole and sums over all 32 lanes, so it is not affected. The same split broke the gated residual's norm kernel later (below).
- **Activations by tile for the tiled matrix kernel** (2026-11-28; `gemm.blocked_shape`, `gemm.blocked_linear`).
  - **What was slow.** A step of the tiled kernel copies 32 inputs of each of its block's 256 rows into local memory. In a (rows, inputs) array those are 256 pieces of 64 bytes, a row apart. With many inputs and few outputs, few workgroups share each piece and the kernel waits for memory: 12 TFLOPS for the gated residual's down projection (10,240 → 320), 25 for the DeltaNet and attention output projections (6,144 → 2,560), against 40 for the layers with many outputs. Diagnostics on the output projection's shape (`gemm.DIAG`): 19 TFLOPS as it was, 35 when every step reads the same 32 inputs (cached), 43 with no activation loads.
  - **The layout:** (row blocks, 32-input blocks, 256 rows, 32 inputs). A step's activations are 16 contiguous KB, and a workgroup's steps follow each other in memory. Under equal load the output projection takes 1.69 ms instead of 3.32 and the down projection 0.46 instead of 1.18, with the same bits.
  - **Who writes it.** The kernels that produce these activations write them in that layout: the gated residual's norm kernel (`hc`), the DeltaNet block's output kernel (`deltafuse`), and a new kernel for the attention's output times its gate (`qsa_gated_out`). A layer asks for it (`gemm.takes_blocked`) when its call has a fixed number of rows and goes to the tiled kernel; other calls keep rows.
  - **Fewer splits.** Calls of 256-row blocks are split along the inputs until they have 80 workgroups, not 320: the down projection in two splits instead of eight, the output projections and the shared expert's down projection unsplit, its gate and up in two. Most of the kernels that summed the splits are gone. Calls of 128-row blocks (the 27B's) keep 320.
  - **Per call** (profile): dense Q8_0 553 → 427 ms. The down projection 105 → 41, the two output projections 2.65 → 1.64 ms a layer, the shared expert's gate and up with the attention's key and value layers 29 → 17. The norm kernel pays 7 ms for its scattered stores (71 → 78).
  - **A range used through `//` and `%`.** The norm kernel had one range of 256 threads and took the wave as `tid // 32`. When its stores began to use `tid % 32` too, tinygrad split the range into two hardware dimensions, 8 by 32. A hardware wave was then 4 lanes of each of 8 of the kernel's "waves", and its sums across a wave mixed them: 1 / rms wrong by up to 0.3, logits by 17. The timed run with that bug looked like a clean result (1,159–1,168 tok/s); only the comparison of its logits showed it. The kernel now declares the lane as a `WARP` range and the wave as a `LOCAL` one, as the matrix kernels do.
- **The expert router on kernels of its own** (2026-11-28). Besides the experts themselves, an FFN block computes the router's matrix product (2,560 → 512, F32 weights), chooses 10 experts a token with their weights, and builds the tables that group the 20,480 token–expert pairs by expert. In the profile of a 2,048-token call these took 44, 31 and 19 ms.
  - **The matrix product on the tiled kernel** (`gemm`, type F32). A thread reads 16 fp32 weights and rounds them to fp16 as tinygrad's loader does. Calls of whole 128-row blocks go there; smaller ones keep the per-wave loop (`densemm`). It runs unsplit and gives the per-wave loop's logits bit for bit.
  - **Selection in one kernel** (`tinyllm.tg.kernels.moe_select`; calls of 128 rows or more with a fixed number of rows). tinygrad's `pairwise_topk` ranks every logit of a row against the 511 others, and the gather and softmax that follow are two more generic kernels. Here a wave takes a row. Each lane holds 16 logits as integer keys that order as the floats do, and ten rounds each take the wave's largest remaining key, the lowest index first among equal ones: `pairwise_topk`'s order. The kernel also writes the softmax over the selected logits. 0.07 ms against 0.61 (`tools/moe_route_check.py --time`).
  - **Routing tables in chunks** (`moe_mm.route`; calls of 2,048 pairs or more). The table kernel was one workgroup, so it ran on one compute unit: every expert's thread read all the pairs twice. Now a first kernel counts each chunk of 512 pairs per expert, a workgroup per chunk, and the table kernel runs a workgroup per chunk too, each placing its own pairs after those of the chunks before. 0.06 ms against 0.62, the same tables.
  - **One fp16 copy of the block's input.** By the kernel counts, the experts converted the block's fp32 input to fp16 for themselves, and the router and the shared expert shared a second conversion. All now read one copy (`moe_mm._shares_input`), handed to each reader in the shape it uses: asked for a contiguous tensor, tinygrad copies a reshaped one again even when it already is contiguous.
  - **Per call** (profile, complete kernel lists): 1,621 → 1,541 ms in 2,696 → 2,504 kernels. Selection 31.1 → 2.7 ms, routing tables 19.3 → 1.9, the elementwise kernels of the input's size (the conversions among them) 16.2 → 13.7 in 144 → 96 calls. The router's product with the shared expert's gate and up, which the profile now times together (next point): 58.6 → 33.4.
  - **What the profile's kernel times mean.** The GPU does not wait for a kernel to finish before it starts the next one when the next does not depend on it. The router's product (32 workgroups, 40% of the machine) and the shared expert's gate and up now run at the same time: the profile shows the two shared-expert kernels at 0.24 µs each and their time under the router's. The sum over a call is right; the split between neighbours that do not depend on each other is not. This is also why the unsplit product is the better choice although it looks worse alone (below).
- **The mixer on fewer bytes** (2026-11-28). Before each sublayer the gated residual read the 84 MB of fp32 streams three times, in the norm kernel, in the mix and in the residual update, and wrote them once.
  - **The mix from the norm kernel's fp16 tiles** (`hc.MIX_TILES`, `gemm.mix_linear(xn=...)`). The mix is formed in the up projection's stores from the normalised streams. It recomputed those from the fp32 streams, 1.3 million pieces of 64 bytes a row (41 KB) apart. It now reads the fp16 copy that the norm kernel writes by tile for the down projection: a workgroup's 256 rows of 32 dimensions of the four streams are four tiles of 16 KB. In the profile the mix takes 59 ms a call instead of 85; the wall clock went from 1,667–1,669 to 1,627–1,631 ms a call. The mix's inputs are then rounded to fp16, which its readers do to its output anyway.
  - **The residual update in the next norm kernel** (`hc.FUSE_UPDATE`). Where one gated residual follows another inside a block (attention or DeltaNet, then the FFN: 48 of the 96), the second one's norm kernel forms the streams itself, streams + output × gates as `GatedResidual` adds them, writes them and normalises them. The streams are read once for both. 1,637–1,642 → 1,622–1,623 ms a call. The update at the end of a block stays tinygrad's: the block is a compiled function and returns the streams.
  - **Which calls.** Both apply to calls of whole 128-row blocks with a fixed number of rows, the calls whose down projection reads tiles. Smaller calls, the stored golden records' among them, compute as before.
  - **A tensor that is still alive is written out.** tinygrad schedules a block's graph at the block's end, and an intermediate whose `Tensor` object is still referenced then gets a buffer of its own instead of being fused into its reader. My first version of the fused update kept the last sublayer's streams, output and gates in a module variable until the next call: with the fusion switched off and the logits unchanged bit for bit, that alone cost 35 to 50 ms a call, and with it on the fusion measured as a loss. (An attempt weeks earlier had failed the same way.) The parts are now remembered under the id of the tensor the model code carries and dropped when that tensor dies (`hc._remember`, `weakref.finalize`).
  - **Per call:** profile 1,541 → 1,505 ms in 2,507 → 2,363 kernels. The mixer's kernels together 288 → 235 ms: the mix 85 → 59, the norm kernels with the 48 updates they took over and the 48 updates left 162 → 145, the down projection 41 → 31.

### Correctness

- **Per kernel:**
  - grouped experts against tinygrad's decoding of random blocks: 1e-4 of each dot product's size;
  - fused gate/up against an fp16-input float64 reference: 4.7e-4 (the fp16 output) for Q8_0. Types whose block scale is a product (the K-quants, IQ4_XS, IQ3_XXS, IQ2_S) round it to fp16 and then each weight again, up to 2.4e-3;
  - tiled Q8_0: 2e-6;
  - the mixer kernel: fp16 rounding, gates to 1.5e-7.
- **The DeltaNet kernels:**
  - `tools/delta_scan_check.py` (no model) compares every layout of the scan with the recurrence in float64: outputs and state agree to 2e-7 of the largest value, with shared key heads or not, from position 0 and from a later one, and with the state kept at a row inside the call.
  - `tools/deltanet_check.py` compares the block's path on these kernels with tinygrad's, on the model's own blocks with random inputs and states: outputs to 1.6e-4 (they pass through fp16), the recurrent state to 1.5e-6, the convolution state bit for bit, the kept states of a padded call included.
  - In the model, the last logits of the 16,384-token prompt move by 0.80 with the three kernels and by 0.64 more with the scan, the top token unchanged. A control that only changes the call size moves them by 0.93.
  - **The stored records** (`tests/golden/*.amd.json`) prefill in 32-token calls, which these kernels take, so every record moved: by 0.04 logits on the 27B's BF16 file, 0.14 to 0.43 on its quantized files, 0.2 on the small model, against a tolerance of 0.02 between runs on one device. The quantized files move more, probably because decoding quantizes their activations to 8 bits, which turns a last-bit difference into a visible one (not checked further). No file of the 27B or the small model changed a token. Each record was captured again and both captures compared with llama.cpp (`tinyllm oracle reference`, 96 positions per file):

    | File | Same top token, before and after | Mean / max log-probability difference before | After |
    |---|---|---|---|
    | 27B `BF16` | 96 / 96 | 0.0010 / 0.0100 | 0.0010 / 0.0098 |
    | 27B `Q8_0` | 95 / 96 | 0.0047 / 0.0786 | 0.0049 / 0.0749 |
    | 27B `UD-Q8_K_XL` | 95 / 96 | 0.0042 / 0.0665 | 0.0040 / 0.0659 |
    | 27B `UD-Q4_K_XL` | 95 / 96 | 0.0114 / 0.1081 | 0.0115 / 0.1081 |
    | 27B `Q6_K` | 95 / 96 | 0.0106 / 0.1400 | 0.0107 / 0.1400 |
    | 27B `Q4_K_M` | 95 / 96 | 0.0097 / 0.1109 | 0.0099 / 0.1188 |
    | 27B `Q4_0` | 95 / 96 | 0.0080 / 0.0848 | 0.0081 / 0.0848 |
    | 0.8B `Q8_0` | 96 / 96 | 0.0155 / 0.1595 | 0.0144 / 0.1595 |
    | 0.8B `UD-Q4_K_XL` | 96 / 96 | 0.0160 / 0.1260 | 0.0170 / 0.1260 |

    The one disagreement of each quantized 27B file is the same near tie before and after.

    Flash-Next's two records moved by 1.4 and 1.6 logits, its usual noise (the 512-expert router amplifies small differences). `UD-IQ4_XS` kept its tokens; `UD-Q4_K_XL` changed one, at a near tie in the 3,128-token prompt. Both agree with llama.cpp a little better than before (the table below).
- **Activations by tile:**
  - `tools/tile_layout_check.py` (no model): the tiled kernel gives the same bits from rows and from tiles (Q8_0 and Q6_K, wide and narrow output blocks, 128- and 256-row blocks); the norm kernel and the DeltaNet output kernel write the same bits in both layouts; the attention's gated output kernel matches tinygrad's expression bit for bit.
  - In the model: with the row layout's splits, the last logits of the 16,384-token prompt are the same bit for bit as with rows, and rows give the bits they gave before this work. The new split choice moves them by 0.85, with the same top token (sums in another order).
- **The router's kernels:**
  - `tools/moe_route_check.py` (no model): the selection kernel picks exactly `pairwise_topk`'s experts, in its order, on logits with equal values, signed zeros and a wide range, at four sizes; its weights are tinygrad's bit for bit and within 1e-7 of float64; rows of NaN, infinities and random bits still select 10 distinct experts in range. The tables built in chunks are the one-workgroup kernel's, entry for entry (even, skewed and padded selections), and the pairs by expert are NumPy's stable sort. `tools/prefill_kernel_check.py --type f32` checks the tiled kernel's F32 path against NumPy. Both run in the GPU test suite (`moe-route`, `f32-prefill`).
  - In the model, 4,096 tokens of code, each change alone against none: the shared copy and the chunked tables give the same logits bit for bit; so does the selection kernel when tinygrad computes the weights from its experts (`moe_select.WEIGHTS = False`), so the experts are the old path's on real data. With the kernel's own weights the logits move by up to 1.4, and by 0.79 on the 16,384-token prompt, with the same top token: the weights are the same numbers to the last bit when computed alone, but the kernel that sums the experts' outputs is then built from another expression and sums in another order.
  - The router's product on the tiled kernel gives the per-wave loop's logits bit for bit on the 16,384-token prompt.
  - Against llama.cpp b12675, the `UD-Q4_K_XL` golden prompts captured with 2,048-token prefill calls (the stored records use 32-token calls, which none of these kernels take): 105 / 112 same top token before (mean / max log-probability difference 0.051 / 0.81), 106 / 112 now (0.051 / 0.81). The stored records are unchanged.
- **The mixer on fewer bytes:**
  - `tools/tile_layout_check.py` (no model): the mix from the fp32 streams and from the fp16 tiles both match NumPy to 3e-7 of the largest value, and differ from each other by 2.6e-4 of it (the streams' rounding to fp16). The norm kernel with the update writes tinygrad's own sum bit for bit; its normalised streams, gates and 1 / rms differ from the plain kernel's on those streams by one unit in the last place (the compiler fuses the two kernels' multiplies and adds differently).
  - In the model, the 16,384-token prompt: with both off, the logits are the router step's bit for bit. With both on they move by 0.64 with the same top token (the mix alone 0.80, the update alone 0.66).
  - Against llama.cpp b12675 with 2,048-token calls. The golden prompts hold one prompt long enough to reach these kernels: 106 / 112 same top token before, 105 / 112 with the mix from tiles (a near tie on that prompt, which has flipped both ways across earlier changes). Six longer prompts (four of code, two of prose, 2,120 to 3,346 tokens, 16 steps each):

| Run | Same top token as llama.cpp | In its top 8 | Mean / max log-probability difference |
|---|---|---|---|
| before | 88 / 96 | 95 / 96 | 0.131 / 2.60 |
| the mix from tiles | 90 / 96 | 96 / 96 | 0.115 / 2.60 |
| and the update in the norm kernel | 89 / 96 | 96 / 96 | 0.133 / 2.63 |

  - The stored golden records are unchanged (their calls are 32 tokens). The CPU suite (197) and the 61 GPU tests pass.
- **Rounding noise:** the mixer kernel moves the logits of a 2,600-token prompt by up to 1.4. A control that only changes the chunk size moves them as much (0.8–1.1).
- **The grouped attention and a key-less tile:** a union tile can hold no key of one of the group's rows. When that is a row's first tile, its running maximum is still −∞, and −∞ − (−∞) made the rescaling NaN. The checkpoint test's repetitive prompt (one code snippet repeated) hit it: NaN logits, garbage tokens. The maximum is now clamped at −1e30. `tools/qsa_check.py` has a case with hand-made disjoint lists that fails without the clamp. The check tools had let a NaN pass, because Python's `max(0.0, nan)` is 0.0; they now use `np.maximum`.
- **Determinism and prompt reuse:** the same prefill three times gives identical logits. `tools/checkpoint_check.py` with 1,024-token chunks (static, symbolic and short-call programs) resumes bit for bit, at 2,304 and at 2,609.
- **The model against llama.cpp b12675** (the golden prompts, re-captured):

| Run | Same top token as llama.cpp | Mean / max log-probability difference |
|---|---|---|
| `UD-Q4_K_XL`, and the DeltaNet kernels | 108 / 112 (96.4%) | 0.042 / 0.76 |
| `UD-Q4_K_XL`, and fp16 expert outputs | 107 / 112 (95.5%) | 0.045 / 0.76 |
| `UD-Q4_K_XL`, block experts and key-split attention | 107 / 112 (95.5%) | 0.047 / 0.79 |
| `UD-Q4_K_XL`, F3 so far | 108 / 112 (96.4%) | 0.046 / 0.65 |
| `UD-Q4_K_XL`, F2 | 105 / 112 (93.8%) | 0.065 / 0.86 |
| `UD-IQ4_XS`, and the DeltaNet kernels | 110 / 112 (98.2%) | 0.040 / 0.67 |
| `UD-IQ4_XS`, and fp16 expert outputs | 110 / 112 (98.2%) | 0.045 / 0.80 |
| `UD-IQ4_XS`, block experts and key-split attention | 110 / 112 (98.2%) | 0.038 / 0.48 |
| `UD-IQ4_XS`, F3 so far | 109 / 112 (97.3%) | 0.036 / 0.45 |
| `UD-IQ4_XS`, F3 before its grouped experts | 111 / 112 (99.1%) | 0.036 / 0.62 |
| `UD-IQ4_XS`, F2 | 108 / 112 (96.4%) | 0.047 / 0.65 |

### Speed

One 2,048-token pass of `UD-Q4_K_XL` (`tools/profile_step.py --kind prefill --chunk 2048`), kernel time:

| Step | ms per pass | Note |
|---|---|---|
| F2 kernels, grouped experts | 3,462 | experts 764, generic 1,325, dense Q8_0 674, attention 314, router 230, DeltaNet 138 |
| + BEAM-tuned generic kernels | 2,888 | generic 1,325 → 781 |
| + router on the matrix cores | 2,702 | |
| + mixer kernel | 2,520 | |
| + fused gate/up | 2,457 | |
| + tiled Q8_0 | 2,406 | dense Q8_0 608 → 517 |

The same pass of `UD-IQ4_XS`:

| Step | ms per pass | Note |
|---|---|---|
| the kernels above, its experts one row per pair | 11,937 | |
| + its expert types grouped | 2,919 | |
| + IQ4_XS vector loads | 2,647 | |
| + Q6_K on the tiled kernel | 2,549 | Q6_K 396 → 268 |

At 128-token chunks the pass was 717 ms for 128 tokens, 11.6 times as long per token. Experts at 2,048 rows against the one-row kernels, one call at the model's shapes (`tools/moe_mm_check.py`): Q4_K gate 5.0 against 83.5 ms, Q5_1 down 7.1 against 110.5 ms.

Wall clock, `UD-Q4_K_XL`, 2,048-token calls:
- `tinyllm bench --chunk 2048 --prefill 10240 --ctx 24576` (positions 0 to 20k): 2,362 ms per call, 867 tok/s, before the grouped attention.
- **16,384 tokens of code** (tinyllm's sources; a scratch script timing each call and checking the last logits are finite): grouped attention and fused mix 795–797 tok/s, calls 2,096 → 2,400 ms. One-row attention: 773 tok/s, calls 2,198 → 2,486 ms.
- **The same with the block expert kernel: 861 tok/s,** calls 1,911 → 2,212 ms, the same last-row top token. With the key-split grouped attention too: **878 tok/s**, calls 1,921 → 2,151 ms. With fp16 expert outputs and the n-gram rows off the critical path, measured with the server's wait mode: **1,016 tok/s**, calls 1,810 → 2,076 ms, 0.02 s outside the calls. `UD-IQ4_XS`, the same prompt: 995 tok/s, calls 1,863 → 2,119 ms.
- **With the DeltaNet kernels** (2026-11-27, the same prompt and wait mode, two timed runs per setting, one process after the other): 1,010–1,013 tok/s before, 1,058–1,061 with the three kernels around the scan, **1,089 tok/s** with the scan's rows on four lanes too. Calls 1,760 → 1,932 ms (before: 1,900 → 2,076).
- **With activations by tile** (2026-11-28, the same prompt, two timed runs per setting): rows as before 1,067–1,078 tok/s; tiles with the rows' splits 1,127; tiles and the new splits **1,166–1,173 tok/s**, calls 1,624 → 1,799 ms.
- **The server with the DeltaNet kernels** (one run each, my own instance): in the agent benchmark a step with about 1,310 new tokens, one call padded to 1,536, waits 1.71–1.72 s for its first token, and a step with about 1,550, padded to 1,792, waits 1.89–1.92 s. Before: 1.89–1.91 s (1,514–1,533 new tokens) and 2.11–2.12 s. The first step's 4,402 tokens: 4.88 → 4.58 s. The SDK check's 6,412-token prompt: 968 → 1,045 tok/s, first token 6.62 → 6.14 s.
- **With the router's kernels** (2026-11-28, the same prompt, two timed runs per setting, one process after the other): as before 1,171–1,175 tok/s (calls of 1,743–1,749 ms on average); the router's product on the tiled kernel 1,187–1,190; with the selection kernel 1,206–1,210; with the tables in chunks 1,229–1,231; with one fp16 copy **1,234–1,235 tok/s** (1,659–1,660 ms, calls from 1,537 to 1,708).
- **The server with activations by tile** (2026-11-28, one run, my own instance): in the agent benchmark a step with about 1,400 new tokens, one call padded to 1,536, waits 1.56–1.65 s for its first token, and a step with about 1,600, padded to 1,792, waits 1.74–1.78 s (the day before 1.71–1.72 and 1.89–1.92 s). The first step's 4,402 tokens: 4.58 → 4.18 s. The SDK check's 6,412-token prompt: 1,045 → 1,141 tok/s, first token 6.14 → 5.62 s. The 27B's Q8_0 file, whose output projection takes tiles in blocks of 128 rows, is unchanged: 372.5 tok/s (`tinyllm bench --chunk 128 --prefill 2048`).
- **With the mixer on fewer bytes** (2026-11-28, the same prompt, two timed runs per setting, one process after the other): as before 1,222–1,231 tok/s (1,663–1,676 ms a call), now **1,263 tok/s** in both runs (1,621 ms, calls from 1,502 to 1,672).
- **The server with the router's kernels** (2026-11-28, one run, my own instance): in the agent benchmark a step with about 1,310 new tokens, one call padded to 1,536, waits 1.49–1.52 s for its first token, and a step with about 1,550, padded to 1,792, waits 1.67–1.69 s (earlier that day 1.56–1.65 and 1.74–1.78 s). The first step's 4,402 tokens: 4.18 → 3.95 s. The SDK check's 6,412-token prompt: 1,141 → 1,199 tok/s, first token 5.62 → 5.35 s. The start that rebuilt the programs was ready after 860 s, the next one after 60 s with its 33 programs loaded. The 27B has no layer these kernels take (its only two-dimensional F32 tensors are the DeltaNet convolutions), and its golden records pass unchanged.
- **The server with the mixer on fewer bytes** (2026-11-28, one run, my own instance): the SDK check's 6,412-token prompt 1,199 → 1,220 tok/s, first token 5.35 → 5.26 s. In the agent benchmark the first step's 4,402 tokens wait 3.88 s instead of 3.95; the steps with one padded call are within the run-to-run spread (1.51 s for about 1,310 new tokens, 1.66–1.68 s for about 1,550). The start that rebuilt the programs was ready after 820 s.
- **The 27B with the DeltaNet kernels** (`tinyllm bench --chunk 128 --prefill 2048`, two runs per setting, `TINYLLM_KERNELS` without and with `delta`): `UD-Q4_K_XL` 355–359 → 372–373 tok/s, Q8_0 364–365 → 373–374. Decode is unchanged (12.44 and 7.94 tok/s): the kernels take prefill calls only.
- The timed text is frozen since 2026-11-26 (a copy of the sources and tools of that day), so that source edits do not change the prompt. Host time varies with the page cache: the n-gram table (28.8 GB) does not stay cached beside the GPU's 111 GB. A profile of real code at positions 2k to 4k: 2,200 → 2,060 ms summed, experts 782 → about 575.

**Expert routing on real code** (a dump of every layer's selections over the 8 calls of those 16,384 tokens). Per layer, about 437 of the 512 experts get rows:

| Rows of an expert | Experts per layer | Rows per layer |
|---|---|---|
| 1–16 | 211 | 1,311 |
| 17–48 | 120 | 3,538 |
| 49–128 | 68 | 5,205 |
| 129–512 | 37 | 8,702 |
| more | 2.6 | 1,724 |

The busiest expert of a layer gets up to 1,425 of the 2,048 tokens. Tiles of 128 rows: 503 per layer; tiles of 64 and a rest: 624. `tools/moe_mm_check.py --routing` replays such a dump layer by layer, and `--skewed` approximates it (shares falling as 1 / rank).

**The block kernel against the per-wave kernels**, layer 60 of the dump at the model's shapes, ms per call (`moe_mm_check.py --routing`):

| | Fused gate/up (Q4_K) | Down (Q5_1) |
|---|---|---|
| per-wave kernels, tiles of 64 and a rest | 10.09 | 7.06 |
| block kernel | 7.59 | 6.19 |

What the block kernel's time is (per-kernel profile, the same layer; diagnostics in `moe_mm.WIDE_DIAG`):

| | Gate + up | Down |
|---|---|---|
| as built | 6.58 | 5.13 |
| every tile reads expert 0's weights (cached) | 5.26 | 3.15 |
| constant activations | 5.29 | 5.22 |
| constant weights (no loads, no decoding) | 3.66 | 2.60 |

The matrix work alone runs near the chip's peak. Weight traffic still does not overlap it: 2 ms of down's 5.1, 1.3 of gate/up's 6.6. Decoding costs 1.6 ms in gate/up.

A first measurement gave the grouped attention 924 tok/s against 757, with calls getting faster as the context grew. That was the NaN bug below: once some rows' hidden states were NaN, their expert routing degenerated onto a few experts, and the pass got cheaper. A kernel bug can make a pass faster. Check the output of every timed run. The two-wave split of the output dimensions was chosen in that run (against 4: 888, and 1: 862) and not re-measured.

`tinyllm bench`'s synthetic prompt (token ids 1,000 to 1,999, repeated) is a poor stand-in for this model, because block selection and expert routing follow the content. Measure Flash-Next on real text.

In use:
- **A 16,384-token prompt in a session:** 20.5 s, 800 tok/s, of which 0.9 s is host time (the n-gram rows).
- **The agent benchmark** (as in section 13): prefill 560–650 tok/s; 26 s waiting for the first token over steps 1–7, against 111 s with F2 and 300 s with F5. Decode is unchanged at 25–27 tok/s.

### Tried, not kept

Measured in the model at 2,048 rows unless noted:
- **Q8_0 per-wave loop:** 128-row token tiles (887–991 ms against 654); activations shared in local memory (930–1,773); no local-memory reservation (752); weights a step ahead (654 against 608 in a later run); other compiler interleaving hints (no gain).
- **Expert kernel:**
  - 4 waves per workgroup: no gain.
  - Gathered activations shared in local memory: 1,021–1,609 against 752.
  - One group per step: 1,093.
  - Fused gate/up with two output tiles per matrix: 1,043. With one, the default, 746.
  - Only 64-row (or 48, 32) tiles: 1,117–1,977.
  - Weights a step ahead: 796.
- **Tiled kernel:** two buffers with the next step held in registers: 20 against 25 TFLOPS. Two blocks per step: 22. 256 × 256 blocks: 24.7. One buffer allocated as two: 20, so local memory limits how many workgroups share a compute unit.
- **4,096-token passes:** 2,305 ms per 2,048 tokens, 6% better, for twice the activation memory.
- **A tiled expert kernel** (as the tiled gemm: a workgroup of 4 waves per tile and 64 outputs, activations and weights staged in local memory). Where the per-wave loop's time goes, at the model's shapes with uniform routing, ms per layer: 16.3 in all; 14.8 with constant activations; 12.6 with every expert reading expert 0's weights from the cache; 9.9 with constant weights; 8.7 with both constant. So the weights' memory traffic (3.7) and conversion (2.8) add to the matrix work instead of overlapping it, at 200–256 VGPRs (6–7 waves per SIMD).
  - fp16 weights staged per 32 inputs: gate/up 13.6 against 9.2 ms. Each step reads 16 bytes from each of 128 weight rows, so the caches thrash. The next step's raw words held in registers: no change.
  - Raw blocks staged per 256 inputs (128 for 32-weight blocks), each wave dequantizing from local memory: gate/up 10.2 against 9.2, down 6.3 against 7.1.
  - In the model, the faster kernel per tile class: 52 ms less in a profile, 12 ms more in wall clock (2,374 against 2,362 ms per call, synthetic prompt). On code it gave 912 against 897 tok/s, but that run had the grouped attention's NaN bug, so it is not evidence either way. Removed.
- **Barriers that fence local memory only.** tinygrad's barrier is a workgroup-scope fence, which in WGP mode waits for every outstanding global load. A barrier fenced to local memory changed neither the tiled expert kernel nor the tiled gemm, pipelined or not (gemm 25–36 TFLOPS either way). The block kernel needs it: it keeps prefetched loads in flight across its barriers.
- **The residual update inside the next sublayer's norm kernel.** That kernel would read the previous streams, the sublayer's output and its gates, and write the new streams, to save one 84 MB read of the streams per sublayer. Only 48 of the 96 updates reached it: elsewhere the streams pass through other operations first. The calls got about 100 ms slower (971 against 1,016 tok/s), and the logits moved by up to 0.75. tinygrad already fuses the update with the sublayer's last operations, and splitting them makes the output be written out. Removed. Since 2026-11-28 the 48 updates inside a block are in the norm kernel after all: what had made the output be written out was a reference my hook kept to it ("The mixer on fewer bytes").
- **Towards the block kernel** (2026-11-26, `moe_mm_check.py` on layer 60 of the routing dump, fused gate/up then down, ms per call; before: 10.09 and 7.06):
  - A gate/up pair per wave in 128-row tiles needs 128 accumulator registers. With tinygrad's local-memory reads, the compiler hoisted every tile's reads ahead of the matrix work and spilled: 15.8. With the reads inside the tile blocks: 10.16 and 7.04. With weights loaded a step ahead and barriers fencing local memory: 9.48 and 6.44. Two or three steps ahead: no change. Replaced by one matrix per wave and block-sized weight loads.
  - Experts of at most 48, 32 or 16 rows kept on the per-wave kernels: gate/up 8.3–9.5 against 7.6. On those experts alone, kernels of their own height (16, 32 or 48 rows): 7.9 against 4.9 on the per-wave kernel.
  - No activation load for rows past a tile's count (a branch per load): gate/up 7.58 → 8.72.
  - Activations two or three steps ahead: within noise (7.47–7.58, 6.12–6.28); kept at two.
  - Down's rows loaded whole (its 20 steps unrolled, 80 registers of weights): 10.5 against 6.3.
  - Workgroups of 2 or 8 waves: slower than 4.
  - The per-wave kernel at 4 groups (128 inputs) per step: small gate/up tiles 4.92 → 4.49 ms, the model 820 → 827 tok/s. It is the default for that kernel, which now runs only with `SMALL_MAX` above 0 or `WIDE_TILES` off.

- **Other layouts of the DeltaNet scan** (2026-11-27, ms per 2,048-token call in the profile; tinygrad's layout 150, a state row on four lanes 101):
  - A row on one lane, so that no sum crosses lanes: 682. The lane's 128 state values and the 128 keys do not fit the wave's registers, and the compiler spills them.
  - A row on two lanes: 104. On eight lanes: 170, slower than tinygrad's 32, for a reason not found (by operation count it should lie between four lanes and 32).
  - tinygrad's layout with its sums across the wave done by lane permutes in place of swizzles: bit-identical and no faster. The swizzles themselves are not what limits the scan.

- **Towards activations by tile** (2026-11-28):
  - A copy from rows into tiles by a generic kernel: 0.7 ms for the output projection's 25 MB and 1.1 ms for the down projection's 42 MB, more than the layout saves (1.6 and 0.7 ms). The producing kernels write tiles instead.
  - Tiles ordered by input block first, then row block: the same speed under equal load (1.697 against 1.693 ms). A first measurement gave it 1.40 against 1.84: the slow copy that preceded it let the GPU cool, and the kernel after it ran faster. Compare kernels with the same kernels around them.
  - Tiles for the layers with many outputs (qkv, 2,560 → 10,240): 2.91 against 2.97 ms, within noise. They keep rows; their input is shared by several layers.

- **Towards the router's kernels** (2026-11-28):
  - The router's product in four splits (128 workgroups instead of 32). Alone the kernel is faster, 0.34 against 0.68 ms in the profile, but the sum of its splits reads 16 MB (0.20 ms), and the unsplit kernel leaves room for the kernels after it: 1,730–1,733 ms a call in four splits, 1,721–1,725 unsplit.
  - Summing the splits inside the selection kernel instead: not built. The same reasoning gives it no more than the unsplit kernel has.
  - Activations by tile for the router: not built. The router shares its input with the experts and the shared expert, which read rows.

- **Towards faster experts** (2026-11-28; nothing kept). A scratch benchmark runs one FFN block's expert chain as the model does, with random weights and the real routing of four layers of the dump: 11.5 ms a layer, of which gate/up 6.7–6.8, down 4.2 and the weighted sum 0.68. The profile's per-layer figures in the model are the same (6.4, 4.2, 0.67).
  - **Time follows bytes.** Per layer the chain moves about 1.9 GB: 1.34 GB of weights (437 experts of 3.07 MB), the gate's fp32 scratch buffer (52 MB written, 52 read), the SwiGLU outputs (26 and 26), the down projection's outputs (105 written, 105 read by the sum) and the sum (21). Each stage's time is its bytes at about 140 to 185 GB/s (gate/up 0.93 GB in 6.7 ms, down 0.67 GB in 4.2, the sum 126 MB in 0.68); the chip reads 241 GB/s at best. The scalar instructions of the loop do not matter, nor does the number of workgroups.
  - Workgroups past the routing's tiles made to read cached data: no change (11.48 against 11.52 ms).
  - The gate's scratch buffer in fp16: gate/up 6.71 → 6.45 ms. It rounds the gate before the SwiGLU, so it was not taken further.
  - The scratch buffer laid out by workgroup, each one's rows contiguous: 7.21 against 6.71, padded between blocks or not. The same bits.
  - Workgroups of 8 or of 2 waves: gate/up 7.13 and 7.08 against 6.71, down 4.28 and 4.33 against 4.16.
  - The tile's row count as one scalar per wave, so that the tests on it are a scalar compare and a branch (they compile to a mask, a select, a compare and a branch): the loop falls from 2,148 to 1,710 instructions and takes the same time.
  - No activation load for the row groups a tile does not have, behind those scalar tests: gate/up 7.19 and down 4.55, slower, as the per-lane version was.
  - **Gate and up in one wave on tiles of at most 64 rows** (which leave half of the accumulator registers free; a class of tiles of their own, about 70% of the tiles and a third of the rows): no scratch buffer for those rows and one staging for both. Gate/up 6.78 → 6.50 ms in the benchmark (6.54 at 48 rows, 6.66 at 32), the same bits for every weight type. In the model: 1,227–1,228 tok/s against 1,225–1,232 without it, the same logits. Not kept. Splitting the tiles into the two classes costs nothing by itself.
  - Two groups of 16 outputs per wave for the down projection on those tiles: 4.23 against 4.14.
  - The down projection's outputs stored with the non-temporal hint: 4.55 against 4.19.
- **Towards the mixer on fewer bytes** (2026-11-28):
  - A norm kernel whose threads take 8 consecutive values of each stream (vector loads, 10 waves a token, a tenth of the memory instructions): 0.75 ms a call alone against 0.82, but 1,696–1,698 ms a prefill call in the model against 1,627–1,631. The update fused into it took 30 ms back; the kernel kept its original layout.
  - The mix written a second time in fp16 by the mix kernel, for the layers that convert it (84 of the 96 conversion kernels go, the same bits): the mix kernel 59 → 71 ms, and 1,648–1,653 ms a call against 1,622–1,623. Writing the copy there costs more than the conversions did.

- **Towards a faster DeltaNet** (2026-11-28; nothing kept). The block takes 157 ms a call, 103 of them the scan, against Gufo's 97 for its one fused kernel.
  - **The scan is bound by its arithmetic.** It is 768 waves that each step through the call's 2,048 tokens, about 250 instructions a token, and together they keep the chip's vector units busy. Per state value and token the exact recurrence needs a product for the key dot, one for the query dot and a multiply and a multiply-add for the update. A benchmark of the scan at the model's sizes (2.72 ms a call; 2.86 in the model): without the query dot 2.03 ms, without the decay multiply 2.61 (both wrong results). The compiler already pairs the update's multiplies two to an instruction and does not pair the dots' multiply-adds.
  - A lane's keys and queries loaded 8 values at a time: 2.74 against 2.72, the same bits. The compiler had merged the scalar loads already.
  - Compiled without the `s_delay_alu` hints (a quarter of the loop's instructions, with the waits): 2.78 against 2.74. The hints are free.
  - Keeping the state divided by the running product of the decays would remove the decay multiply: worth 4% of the scan by the figure above, for a rescaling that depends on the data. Not built.
  - **The output norm and gate in the gate projection's stores** (the tiled kernel forms the block's gated output and the projection's 50 MB are never written; a small kernel gives the 1 / rms): the same logits bit for bit, and 1,663–1,672 ms a call against 1,620–1,627. In the profile the projection with those stores takes 3.12 ms a layer, where the plain projection takes 1.75 and the output kernel 0.63. The tiled kernel has two workgroups per compute unit, and the per-value work at its end (a SiLU, three multiplies, two scattered loads and a store) runs at that low occupancy. The gated residual's mix pays the same price and is still worth it, because it replaces 168 MB of traffic, not 100.

### What limits it, and what is next

Gufo's 2,048-token pass on this machine takes 1,116 ms. Its own breakdown (`docs/internal/models/qwen3.8-flash-next/EXPERIMENTS.md`, 2026-10-24) is in the first column. The others are tinyllm's, from profiles of real code at positions 2k to 4k (`tools/profile_step.py --text --max-context 10240`; kernel time with a sync after each kernel): with the per-wave expert kernels (2,200 ms in all), and with the block kernel (2,060 ms):

| | Gufo, ms | tinyllm, per-wave experts | tinyllm, block experts |
|---|---|---|---|
| routed experts | 388 (its "DRAM floor about 305") | 782 | about 575: down Q5_1 224, up with SwiGLU 148, gate 138, down Q8_0 32, route 29 |
| dense GEMMs | about 350 (F16) + 62 (W8A8) | 621 | 569 (Q8_0, the mix's up projection included) |
| tinygrad's generic kernels | in its rows above | 406 | 437 (residual updates 89, expert combine 61, DeltaNet's conv and norms, router top-k) |
| DeltaNet scan | 97, its whole DeltaNet | 147 | 148, the scan alone |
| attention | 34 (depth 0) | 173 | 171 |
| residual mixer combines | 133 | 70 | 72 (the norm kernel) plus part of the generic row |

The used experts' weights are about 1.34 GB per layer on code (437 experts of 3.07 MB), about 270 ms per pass at 240 GB/s.

`--max-context` must exceed the warm-up's four calls (with `--context 2048 --chunk 2048`, more than 8,192).

`profile_step.py` without `--text` warms up with four calls that advance the position, so its "`--context 2048`" profiles in this section were at positions 8,192 to 10,239. With `--text` it resets after warming up.

Gufo measured the chip's F16 matrix peak at 54.8 TFLOPS. Kernels run power-limited near 1.9 GHz, so their ceiling is about 39 to 48 TFLOPS.

On 16,384 tokens of code a call now takes 1,500 to 1,670 ms, and the host adds almost nothing: 1,263 tok/s (2026-11-28, with the mixer on fewer bytes; 1,234–1,235 with the router's kernels and 1,166–1,173 with activations by tile earlier that day; 1,089 with the DeltaNet kernels the day before; 1,016 before those). 1,500 tok/s needs about 1,370 ms per call.

The profile of 2026-11-28 (real code at positions 2k to 4k) sums to 1,505 ms (1,541 before the mixer's changes, 1,621 before the router's kernels; 1,779 the day before, with the DeltaNet kernels and activations in rows):

| | ms | Kernels |
|---|---|---|
| routed experts | 515 | down Q5_1 180, up with SwiGLU 153, gate 147, down Q8_0 26, the Q5_K layer 7, routing tables 2 |
| dense Q8_0 | 380 | layers with many outputs (qkv, gate, attention query), the output projections and the small layers 287, the mix's up projection 59, the gated residual's down projection 31 |
| DeltaNet | 157 | scan 103, output norm and gate 23, values 18, queries and keys 13 |
| sparse attention | 153 | attention 131, gated output 12, selection and scores 10 |
| tinygrad's generic kernels | 129 | the residual updates at the ends of the blocks 41, expert combine 32, fp16 conversions of the sublayers' inputs 13, the rest under 8 each |
| mixer norm and gates | 104 | with the residual update of the sublayer before 65 (48 calls), without 39 (48 calls) |
| router | 36 | the matrix product, timed together with the shared expert's gate and up, 33; selection 3 |
| other dense kernels | 30 | small F32 layers on tinygrad's kernel 13, BF16 layers 14, a Q8_0 layer on the per-wave loop 3 |

Kernels that do not depend on each other overlap on the GPU, and the profile then puts their time under one of them ("What the profile's kernel times mean", above). The rows are sound as groups; a single kernel's figure can hold a neighbour's time.

The layers with many outputs run at about 40 TFLOPS, the output projections at 39: near what the matrix loop reaches without loads (39–47). What is left in the dense layers is the mix's up projection, whose time is the 84 MB of streams its stores read.

Left in the mixer: 235 ms a call against Gufo's 133 for its combines. The 48 updates at the ends of the blocks cannot move into the next block's norm kernel, since each block is a compiled function that returns the streams. The sublayers' fp32 inputs are still converted to fp16 by a generic kernel, once per sublayer (13 ms).

**The experts against Gufo's, on the same footing** (2026-11-28). Gufo's 388 ms is for its benchmark prompt, on which 66% of the experts of a layer are active and a 2,048-token pass has 408 tiles a layer (its `EXPERIMENTS.md`). On code 85% are active (437 of 512) and a pass has 503 tiles. The kernels' time follows the weights they read ("Towards faster experts", above), so Gufo's kernels would take about 480 to 500 ms on this routing, against 515 here. What separates this pass from Gufo's is then mostly elsewhere: the mixer, the dense layers and the generic kernels.

**Bytes moved per pass**, by the same accounting: the experts' weights 64 GB; the mixer about 42 GB before its changes (per sublayer, the 84 MB of residual streams read by the norm kernel, by the mix and by the update, and written by the update, and the normalised streams written and read once: 440 MB, 96 times) and about 36 GB since.

The notes below are from the 2,060 ms profile, before fp16 expert outputs and the key-split attention, except where dated:
- **Experts, about 575 ms** against Gufo's 388. Weight traffic does not yet overlap the matrix work: 2 ms per layer of down's 5.1. Down's 640 inputs give a tile only 20 steps, and its 40 workgroups each a short life. Decoding is 1.6 ms of gate/up's 6.6 per layer. fp16 outputs since saved about 2 ms per layer of down's stores and the combine's reads. Next idea: more outputs per down workgroup.
- **Dense Q8_0, 569 ms** (427 since 2026-11-28: what held the slow layers back was how their activations lay in memory, see "Activations by tile") at 29–36 TFLOPS per layer. With constant fragments and no loads the loop reaches 39–47, so loads and dequantisation are not overlapped with the matrix work. The block kernel's tools apply here: weights as the A operand decoded once, block-sized loads a block ahead, barriers fencing local memory only.
- **Generic kernels, about 437 ms** (260 since the DeltaNet kernels): the residual updates (89), the expert combine (61, now over fp16 outputs), DeltaNet's conv window and norms (since in kernels of their own), and the router's top-k. Fusing the residual update into the next norm kernel did not pay (see "Tried, not kept"). Next: a census of these kernels with their sources, to fuse the largest into custom kernels.
- **DeltaNet, 155 ms** (2026-11-27; 295 before its own kernels: tinygrad's scan 148 and about as much in generic kernels around it). The scan is 101 ms with a state row on four lanes. Gufo's whole DeltaNet takes 97 ms, its fused kernel 1.05 µs per row and layer, about 76 ms here. Per token and wave of eight rows the scan now issues about 170 vector operations, 64 of them the state update (a multiply by the decay and a multiply-add per value). Keeping the state divided by the running product of the decays would halve the update, but a decay near zero makes that quotient overflow, so it would need a rescaling step that depends on the data. Not built. Gufo removed a chunked scan as lossy (it flipped a greedy decision on a tool prompt).
- **Sparse attention, 171 ms** at positions 2k to 4k before the key-split change, which took about 60 ms off the late calls. Gufo spends 34 ms at depth 0.

## 15. F4, speculative decoding (2026-11-26)

With the MTP block drafting 4 tokens a cycle, `UD-Q4_K_XL` decodes code at 52–55 tok/s and prose at 34–35, against 26–28 plain; six mixed prompts run at 43.7 tok/s against 27.6. The model's DFlash2 drafter also runs, and is slower than the MTP block on this machine. The verify computes what one-token decoding computes up to rounding, but not bit for bit, so a greedy output can leave plain decoding's where two tokens are nearly tied.

### What was built

- **The block.** Flash-Next's MTP block ships in a GGUF of its own (`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, 2.8 GB, 32 tensors `blk.48.*`). `tinyllm.tg.mtp.load_streams` loads it next to the main model. It is one sparse-attention layer of the main architecture with its own experts, caches and mixers (llama.cpp b12675, `src/models/qwen4exp.cpp`, `graph_mtp`), fed per residual stream by `eh_proj(concat(enorm(embed(token)), hnorm(h)))`, where `h` holds the main model's four streams before its final mixer. Its streams are the next draft's `h`, and its own mixer (`hc_head`) feeds the main model's output head.
- **The cycle** (`UpstreamSession.speculate`), two programs with a host step between them:
  1. Draft. The MTP block runs over the previous verify's K + 1 rows, which writes its attention cache for the accepted ones, and the last accepted row gives the first draft. K − 1 more one-row steps chain the others. Each draft is the largest logit of a Q4_0 copy of the output head.
  2. The host copies the n-gram rows of the last token and the drafts: a position's rows hash the tokens before it, so they are not known until the drafts are.
  3. Verify. The main model runs the K + 1 rows on decoding's kernels and samples each. The drafts are accepted while they equal the sampled tokens.
- **Rolling back.** The DeltaNet blocks keep the state after each row (`SpecDeltaNetBlock`, as on the 27B), and the n-gram layer keeps its convolution's inputs (`SpecPLELayer`). When a draft is rejected, a restore program puts back the state after the last accepted row. The sparse attention and its indexer are indexed by position and need nothing.
- **Prefill** also runs the MTP block over every row, so its attention cache covers the prompt, in all three prefill programs.
- **Server.** `tinyllm serve` finds an `mtp-*.gguf` of the model's architecture next to the model, in the directory above, or in an `MTP` directory of either (`--mtp` names one), and then drafts 4 tokens a cycle.
- **DFlash2.** `--draft-model` takes the model's DFlash2 drafter instead (below). The 27B's drafter code loads it as it is. The session hands it the mean of the four streams at its target layers and splits the cycle in two programs like the MTP's.

Teacher-forced on 128 positions of code, the block's first draft is the main model's own next token at 93.0% of them.

### Speed

Through the runner (`tools/spec_check.py --chunk 2048 --ctx 16384 --tokens 256`, greedy, thinking on for the train prompt only). Each cell is tok/s, with the tokens produced per cycle in brackets; K is the drafts per cycle:

| Prompt | Plain | MTP, K = 3 | MTP, K = 4 | MTP, K = 5 | DFlash2, K = 4 | DFlash2, K = 7 |
|---|---|---|---|---|---|---|
| Python: an ISO 8601 parser with tests | 28.0 | 54.3 (3.64) | 55.4 (4.40) | 60.0 (5.20) | 56.3 (4.18) | 54.8 (5.31) |
| Rust: 0-1 BFS with a test module | 27.9 | 52.8 (3.59) | 52.4 (4.25) | 51.6 (4.64) | 49.0 (3.75) | 46.3 (4.55) |
| TCP against UDP, a table | 27.7 | 46.4 (3.27) | 45.0 (3.70) | 42.5 (3.92) | 44.9 (3.45) | 38.6 (3.81) |
| A train's travel time, thinking on | 28.0 | 48.3 (3.36) | 50.4 (3.98) | 46.9 (4.11) | 37.0 (2.68) | 33.6 (3.15) |
| How a refrigerator works | 26.3 | 35.9 (2.55) | 34.6 (2.87) | 31.2 (2.90) | 33.3 (2.55) | 25.9 (2.52) |
| A short story | 27.5 | 32.7 (2.30) | 34.3 (2.80) | 30.3 (2.80) | 29.9 (2.28) | 24.8 (2.43) |
| All, 1,530 tokens | 27.6 | 43.5 | 43.7 | 41.1 | 39.7 | 34.4 |

- **Plain** is from the K = 4 run, where the n-gram rows were in the page cache. The first run, with rows read from disk, gave 24.9–27.3.
- **K = 4 is the default** for this model: the fastest overall, within 8% of the best on every prompt, and above plain decoding on prose. Code would take 5 (Python 60.0), prose 2 or 3.
- **Runs differ in their text.** A greedy output leaves plain decoding's at a near tie (below), at another token in each run, so the columns do not decode the same 256 tokens.
- **Others:** Gufo reports about 50–55 tok/s with MTP on the same model, and Strata 52.6.

With sampling (temperature 1, top-k 20, top-p 0.95, the same uniform numbers in both runs), K = 4:

| Prompt | Plain | MTP, K = 4 |
|---|---|---|
| Python | 26.2 | 44.5 (3.64) |
| Rust | 26.7 | 49.4 (3.98) |
| TCP against UDP, a table | 27.3 | 41.5 (3.40) |
| A train's travel time, thinking on | 27.2 | 46.2 (3.75) |
| How a refrigerator works | 25.3 | 35.1 (2.90) |
| A short story | 24.7 | 26.4 (2.16) |

The drafts are the block's most likely tokens, so a sampled token that is not the most likely one ends the cycle. Code keeps most of the gain. A sampled story is the weakest case and stays just above plain decoding.

The cycles by accepted drafts show why longer blocks stop paying. With K = 5, 35 of 49 cycles on the Python prompt accepted all five drafts, but only 12 of 88 on the refrigerator prompt, where 26 accepted none.

One cycle at K = 3 (4 rows) takes 68–72 ms:

| Part | ms |
|---|---|
| Draft program: the MTP over 4 rows, 2 chained steps, 3 head reads | 8.6 |
| N-gram rows of 4 positions | 0.2 when cached, 2–4 with about 40% read from disk |
| Verify program | 59 |
| Restore, on the third of cycles that reject a draft | 1.3 |

The verify reads the dense weights once, about 3.5 GB, and each row's experts, about 1.2 GB a row. A plain step takes 36–38 ms. So a row past the first costs about 7.5 ms in the verify and 2.9 ms in the draft, 9.8 ms measured between K = 3 and K = 5, and pays for itself when it is accepted about half the time.

### The server

`tinyllm serve` on `UD-Q4_K_XL` with a 32,768 context, the MTP block found next to the model, 4 drafts, and the programs for partial prefill calls (below):
- **Start.** The first start took 1,090 s, 1,048 s of it compiling and tuning the new programs: the MTP block in prefill, the draft, verify and restore programs, and three more prefill programs. Later starts read them from tinygrad's cache.
- **`tools/openai_sdk_check.py`.** Streaming, reasoning on and off, stop sequences, seeds, the tool-call round trip and two clients at once pass. Its speed request, a sampled story, decodes at 27–28 tok/s, plain decoding's speed (the sampled table above). Prefill of its 6,412-token prompt runs at 968 tok/s.
- **`tools/agent_bench.py`** (8 steps over Python and Rust files, greedy, thinking on, replies of up to 400 tokens):

  | Step | Prompt tokens | Reused | Run | First token, s | Decode, tok/s |
  |---|---|---|---|---|---|
  | 0 | 4,402 | 0 | 4,402 | 4.88 | 58.9 |
  | 1 | 7,220 | 4,352 | 2,868 | 3.72 | 40.6 |
  | 2 | 10,056 | 7,168 | 2,888 | 3.83 | 43.7 |
  | 3 | 11,563 | 9,984 | 1,579 | 2.44 | 42.6 |
  | 4 | 12,853 | 11,520 | 1,333 | 2.28 | 42.1 |
  | 5 | 14,369 | 12,800 | 1,569 | 2.45 | 41.3 |
  | 6 | 17,189 | 14,336 | 2,853 | 3.88 | 40.3 |
  | 7 | 18,488 | 17,152 | 1,336 | 2.35 | 36.0 |

  Decode was 25–27 tok/s in this benchmark before (section 13). A step that adds 1,300 to 1,600 tokens waits 2.3 to 2.5 s for its first token, 585 to 650 tok/s. Before the partial programs the same sizes took 2.4 to 2.5 s (550 to 620 tok/s), and the 2,868-token step 4.08 s instead of 3.72. Such a step is one call of up to 1,536 tokens to the last multiple of 256, where the checkpoint is taken, and a short call for the rest, which costs about 0.33 s whatever its length.

### DFlash2

A DFlash2 drafter for this model was published on 2026-10-19 (`incoai/Qwen3.8-Flash-Next-DFlash2`, GGUF files in `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`; `Q4_K_M` is 362 MB). It has 5 layers of width 2,560 and drafts blocks of 8. It was trained on the mean of the four residual streams at the outputs of layers 5, 15, 25, 35 and 45, which the GGUF names as the inputs of layers 6, 16, 26, 36 and 46, the 27B drafter's convention. llama.cpp cannot drive it yet for lack of that mean.

Its publisher measured it at 1.72–1.80 times plain decoding on code against the MTP block's 1.40–1.45 (SGLang, one GB300, 3 drafts a step). On this machine the MTP block is faster at every setting tried (the table above):
- **A long block costs too much here.** An 8-row verify takes about 97 ms, since every row reads its own experts. With 7 drafts prose falls below plain decoding.
- **At 4 drafts** it accepts about as many as the MTP block on Python (4.18 against 4.40 tokens a cycle) and fewer elsewhere.

On the 27B the opposite holds (research findings, section 5.8): a verify row there costs almost nothing next to the dense weights, so seven drafts pay.

### The n-gram rows

The first runs took 190–255 ms a cycle. 161–179 ms of that was the copy of 64 n-gram rows, 2.7 ms a row. The copy went through a mapping of the table (section 14), where a row that is not in the page cache is read by a page fault, one after the other. Plain decoding paid the same: 81.5 ms a token on new text instead of 36. Memory was not short (33 GiB free).

Calls of fewer than 64 positions now read their rows as section 11 describes: a non-blocking pass, then the thread pool for the rows that are not cached. On new text (sampled at temperature 1, about 40% of rows read from disk) that takes 2–4 ms a cycle, against 20.7 ms through the mapping in the same process. A standalone loop over rows that are in no cache gives 0.7 ms for 64 rows through the pool and 9–14 ms through the mapping. The mapping stays for prefill calls, whose rows a thread writes ahead.

This probably explains the 15 tok/s at the start of a chat in the Jan session of 2026-11-26: a decode step's kernels take 38 ms at positions 1,504 and 5,004 alike.

### Correctness

**Each row against a one-token step** (`tools/verify_rows_check.py`, after a 2,300-token prompt):

| Piece, first row of 4 against 1 row | Difference |
|---|---|
| Every linear and expert layer, the feed-forward | none |
| The DeltaNet sublayer, from the same state | none |
| The mixers | 1e-7 |
| The sparse attention sublayer | 2e-7 |

Through the model those differences do not stay small. The streams agree to 9e-7 after block 3, differ by 7e-4 after block 4 and by 1–3% after the last block. Decoding's kernels round activations to 8 bits, and one value that rounds the other way moves a block's output by about that much. At the logits the first row differs by 0.26 and later rows by up to 1.7. Plain decoding itself agrees with llama.cpp on 107 of 112 top tokens (section 14).

**The rollback.** After `restore_from(j)` every state is the one-token run's after row j up to that rounding, where the next row's lies 75–120% away:
- the DeltaNet states within 1–13% on `UD-Q4_K_XL`, and within 4–8% of the distance to the next row on `UD-IQ4_XS`;
- the n-gram layer's convolution state to 1e-7, except row 2 on `UD-IQ4_XS` at 1e-3, where a rounding had already flipped in the block before the layer.

**Greedy output against plain decoding.** On a 240-token continuation of code, fed the same tokens, the plain step picks another token at 5 positions, with top-2 margins of 0.06 to 0.35. On the chat prompts above, three of six outputs at K = 3 are identical through 256 tokens; at K = 4 all six leave plain decoding's between tokens 24 and 93, at margins of 0.04 to 0.27.

F4's "output identical to plain decoding" is not met. It needs the mixers and the sparse attention to round the same for one row and for several: a mixer kernel that serves any row count, and decoding's attention splits for each verify row.

### Copied blocks (2026-11-27)

An agent's reply often repeats text that is already in the conversation: a file with one change, a quoted function, a tool's output. A copied block proposes that text as drafts. On a reply that repeats a 120-line Python file with a class renamed, `UD-Q4_K_XL` decodes at 112 tok/s, against 62 with MTP drafts alone and 28 plain. Ordinary code and prose are unchanged.

- **The search.** Each cycle, on the host: the latest earlier occurrence of the text's last 16 tokens that has 15 tokens after it (numpy over the session's copy of the token ids; among several occurrences the one that matches furthest back, up to 64 tokens). The 15 tokens that followed are the block.
- **The confirmation.** The MTP draft program runs as in every cycle. The block is taken only if the MTP block's 4 drafts are the block's first 4 tokens. Otherwise the cycle verifies the MTP drafts as usual.
- **The verify** (`_verify_copy`) runs 16 rows, the last token and the 15 copied ones:
  - Dense layers take the matrix-core kernels at 16 rows and read their weights once.
  - Expert layers take the grouped kernels, one read per distinct expert (`moe_mm.MIN_TOKENS` is now 16).
  - Rows are sampled and counted as in any verify, and the rollback is the same: snapshots and restore programs for 15 rows.
  - The program also runs the MTP block over its rows with the tokens the main model chose, so the block's cache stays complete, and leaves the last accepted row's streams for the next draft, as a prefill does.
- **Runs.** After a block whose 15 drafts were all accepted, the next block is verified without the confirming draft. After a block of which fewer than half were accepted, no block is tried for 8 cycles.
- **Server.** On by default for this model with its MTP block (`--copy 15`; `--copy 0` turns it off). The request log shows the tokens that came from copied blocks.

Through the runner, greedy, K = 4 with blocks of 15 (`tools/spec_check.py --copy 15 --repeat`):

| Prompt | Plain | MTP drafts | MTP drafts and copied blocks |
|---|---|---|---|
| The file again with a class renamed, 255 tokens | 27.8 | 62.2 | 112.2 (15 blocks, 225 of 225 drafts accepted) |
| Python | 28.0 | 57.5 | 57.4 |
| Rust | 28.0 | 55.6 | 55.6 |
| TCP against UDP, a table | 28.0 | 47.8 | 47.9 |
| A train's travel time, thinking on | 28.0 | 52.2 | 50.2 (1 block, 5 of 15) |
| How a refrigerator works | 28.0 | 37.1 | 37.1 |
| A short story | 28.0 | 36.4 | 36.4 |
| All seven, 1,785 tokens | 28.0 | | 50.2 |

The MTP column is from the run without copied blocks, whose reasoning on the train prompt took another path. With a match of 8 tokens instead of 16 the Python prompt tried two blocks on a repeated test pattern that then took other arguments (4 of 30 drafts accepted, 56.6 tok/s).

In the agent benchmark through the server (the table under "The server", same steps) the replies are mostly reasoning, and 7 to 68 of a reply's 200 to 400 tokens came from copied blocks. Decode per step went from 36.0–43.7 to 37.3–46.0 tok/s (steps 1 to 7). The first start with the block's verify and its restore programs took 1,495 s; later starts read them from the cache.

A copied block's cycle takes 124 ms for 16 tokens (135 with the confirming draft). Its kernel time, with a sync after each kernel:

| Part | ms |
|---|---|
| Expert layers, grouped | 63 |
| Dense layers, 16 rows on the matrix cores | 34 |
| DeltaNet, a row at a time with snapshots | 14 |
| The confirming draft | 12 |
| Attention, mixers, sampling, the MTP block over the rows | 17 |

On the file-repeating reply the output is identical to plain decoding's over 600 tokens, with and without copied blocks. The block's rows run on the matrix-core kernels, with half-precision activations where decoding's are 8-bit, so in general they differ from one-token steps as prefill rows do.

### What is next

- **Longer copied blocks.** Gufo checks up to 64 copied tokens in one pass. Here 32 rows would spread the dense layers' 34 ms over twice the tokens; a second program would run them once a run is going.
- **One read of an expert per verify.** Rows of one verify often pick the same experts (in a 2,048-token prefill call about 40 of a layer's 512 experts take over a third of the rows, section 14), and each row reads its own copy today.
- **Draft count by request.** Code would take 5 drafts and prose 2 or 3. That needs a pair of programs per count.
- **Identical output.** See "Correctness".

### Partial prefill calls

A call shorter than 2,048 tokens used to go through the program for up to 2,048 tokens, whose kernels compute the padded length (calls of up to 256 had their own program). A profile of one session (kernel time with a sync after each kernel) showed about 800 ms of such a call that did not depend on its length: the DeltaNet scan (137 ms either way), the mixers' projection kernels, tinygrad's generic kernels at 2,048 rows, and the experts, whose 128-row tiles are mostly empty when 454 tokens spread over 400 experts.

The session then got programs for calls of up to 256, 512, 1,024 and 1,536 tokens (`PARTIAL_SIZES`), and a call goes to the smallest that takes it. The two largest have since given way to fixed-length programs ("First-token time of an agent step"):

| Call | Before | With the programs |
|---|---|---|
| 2,048 tokens, the full-pass program | about 2,000 ms, 1,016 tok/s | the same |
| 916 tokens at position 2,600 | 1,557 ms, 590 tok/s | 1,239 ms, 740 tok/s (the 1,024 program) |
| 454 tokens from position 0 | 1,136 ms, 400 tok/s | 705 ms, 640 tok/s (the 512 program) |
| 59 tokens at position 1,505 | 325 ms, 180 tok/s | the same (the 256 program) |

Of the 454-token call's 705 ms the experts take 253: the tiles are still 128 rows wide. Narrower tiles for the smaller programs are the next step there.

### First-token time of an agent step (2026-11-27)

A step of the agent benchmark that adds 1,300 to 1,600 tokens waited 2.3 to 2.5 s for its first token. It now waits 1.9 to 2.2 s.

Where the 2,531 ms of a step with 1,587 new tokens went (the runner alone, a wait after each call; `reused 9,984`):

| Part | Before, ms | Now, ms |
|---|---|---|
| N-gram rows of the first call, read before it starts | 166 | not timed apart |
| Main call, 1,536 tokens to the last multiple of 256 | 1,971 | 1,735 |
| Tail call, 51 tokens, after the checkpoint | 369 | 372 |
| Plan and two copies of the state | 20 | 20 |

- **Fixed-length programs.** The main call ran through the program for up to 1,536 tokens, whose token count is symbolic. tinygrad cannot split or tune a generic kernel along a symbolic axis, so those kernels cost about twice as much per token as in the fixed 2,048-token program, and 1,536 tokens cost as much as 2,048. A stretch between two checkpoint positions is a multiple of 256 tokens, so calls of 768, 1,024, 1,280, 1,536 and 1,792 tokens now have fixed-length programs (`STATIC_SIZES`): 1,536 tokens 1,971 to 1,735 ms, 1,280 tokens 1,795 to 1,459 ms, 768 tokens 1,267 to 1,071 ms.
- **N-gram rows.** A step's new text is rarely in the page cache. Its rows are gathered through the mapping by 64 threads instead of 16 (`GATHER_THREADS`), and the read-ahead for the next call now fetches exactly that call's rows (it assumed another split for a prompt's last stretch and read them twice).

Through the server, `tools/agent_bench.py` (the conversation differs from run to run, so steps are matched by size):

| Step adds | First token before, s | Now, s |
|---|---|---|
| about 1,310 tokens | 2.28 | 1.94–1.96 |
| about 1,550–1,570 tokens | 2.44–2.53 | 2.16–2.20 |
| about 2,870 tokens | 3.72 | 3.56–3.66 |
| 4,402 tokens, nothing reused | 4.88–5.13 | 5.01 |

**What a call costs.** With the fixed-length programs a call takes about 430 ms plus 0.81 ms a token: 1,071 ms for 768, 1,459 for 1,280, 1,735 for 1,536, 2,083 for 2,048. The 430 ms are mostly the experts: a call of a few hundred tokens already touches most of the experts that code uses, about 1.3 GB a layer. The tail call pays them a second time, 0.37 to 0.49 s whatever its length. Taking them away needs a single call per step: a fixed-length call padded past the prompt's end, with the state after its last real row kept and the checkpoint taken inside the call. That was the next step for this number, worth about 0.2 to 0.3 s, and was done the same day (below).

**Start-up.** Every prefill program is traced twice by tinygrad at each start, 100 to 140 s a program even with the kernels cached. A start with ten prefill programs took 1,661 s, two of them new. The programs for up to 1,024 and 1,536 tokens were then dropped (about 280 s), since the fixed-length ones cover the calls that used them. `tinyllm serve` logs the seconds by program. `TINYLLM_STATIC_SIZES=0` starts without the fixed-length programs, about 8 minutes sooner. Since the same day the programs are kept between starts, and only the first start after a change pays this (section 16).

### One call for a prompt's last stretch (2026-11-27)

The stretch that ends a prompt ran as two calls: one to the checkpoint's position, the last multiple of 256, and a short one after it. The short call cost 0.37 to 0.49 s whatever its length, because a call of a few dozen tokens already reads most of the experts. It is now one call, and a step waits 0.12 to 0.28 s less for its first token: 1,382 new tokens 1.98 to 1.77 s, 1,587 tokens 2.15 to 2.01 s.

**How.** Where the model drafts with its MTP block, the fixed-length programs take a count of real rows (`_prefill_padded`). A last stretch of more than 256 tokens runs through the program of the next multiple of 256, and the rows past the prompt's end are padding:

- **Padding rows change no state.** In the DeltaNet blocks they are exact no-ops for the recurrent state (beta 0, decay 1), as tinygrad makes the padding of a symbolic chunk. The convolution states of those blocks and of the n-gram layer are taken from the window at the last real row. The logits and the hidden kept for the first draft are that row's.
- **What they write is never read.** Their attention keys, values and indexer keys lie past the prompt's end, are masked out and are overwritten as tokens arrive, like a rejected draft's. The runner records those positions as belonging to no sequence, so no later request reuses them. They read zeroed n-gram rows, so nothing they compute is infinite.
- **The checkpoint is taken inside the call,** 256 rows before its end, where the real rows of the last block begin at the latest. The DeltaNet scan runs as two kernels, rows before that point and rows after, and the first writes the state into a buffer of its own that the second reads (`gated_delta_rows`): the state at that row costs 3 MB a block and one more launch. The convolution states there are a fixed slice of the window. A copy program moves the kept state into the checkpoint's slot after the call.

A new 512-token program serves stretches of 257 to 512 tokens. A last stretch of at most 256 tokens runs as before, as do sessions without an MTP block and the 27B.

**Speed.** The runner alone, a wait after each call, steps resuming from the previous step's checkpoint (ms to the first token):

| Tokens run | Calls before | ms | Call now | ms | Saved |
|---|---|---|---|---|---|
| 558 | 512 + 46 | 1,180 | 558 padded to 768 | 1,051–1,056 | 124–129 |
| 747 | 512 + 235 | 1,358 | 747 padded to 768 | 1,083–1,085 | 273–275 |
| 1,382 | 1,280 + 102 | 1,980 | 1,382 padded to 1,536 | 1,772–1,775 | 205–208 |
| 1,587 | 1,536 + 51 | 2,153 | 1,587 padded to 1,792 | 2,006 | 147 |
| 2,973 | 2,048 + 768 + 157 | 3,694 | 2,048, then 925 padded to 1,024 | 3,410 | 284 |
| 9,990, nothing reused | four of 2,048, then 1,792 + 6 | 10,422 | four of 2,048, then 1,798 padded to 2,048 | 10,246–10,250 | 172–176 |

- A padded call costs what a plain call of its length costs: 1,587 tokens padded to 1,792 took 1,981 ms, a plain 1,792-token call 1,973 ms. The copy of the kept state takes 5.4 ms, as the copy it replaces did.
- The saving is the short call's 0.27 to 0.52 s less about 0.8 to 0.9 ms for each padding row, so it is smallest where the stretch just passes a program's length (1,587: 205 padding rows) and largest just under one (747: 21 rows).

Through the server, `tools/agent_bench.py`:

| Step runs | First token now, s | Earlier runs, s |
|---|---|---|
| 1,514 to 1,533 tokens (padded to 1,536) | 1.89–1.91 | no step of this size |
| 1,542 and 1,543 tokens (padded to 1,792) | 2.11–2.12 | 2.16–2.20 (1,550 to 1,570 tokens) |
| 2,868 tokens | 3.40 | 3.53–3.66 |
| 3,098 tokens | 3.70 | 3.85 (3,091 tokens) |
| 4,402 tokens, nothing reused | 4.88 | 4.88–5.13 |

The benchmark's conversation differs from run to run, so steps are matched by size where an earlier run had one. A step of about 1,310 tokens took 1.94 to 1.96 s before. Decode in this run: 34 to 56 tok/s by step.

**Correctness.** A padded call cannot give the same bits as the calls it replaces: programs of other lengths round differently, as any two ways of cutting a prompt into calls do.

- `tools/padded_call_check.py` runs the same tokens both ways. The state a padded call ends in, and the state it keeps inside, lie within 10% of one row's change from the state plain calls reach there (2 to 10% by kind of state); two ways of cutting the plain calls lie 4 to 7% apart, and a state one row off would lie 100% away. The logits pick the same token; 13 greedy tokens after the call are the same.
- `tools/checkpoint_check.py --mtp find` runs the runner's four requests. Resuming from the state kept inside a call gives the same bits whatever was decoded in between. Against calls that end at the checkpoint the logits differ by up to 1.6 and the top token is the same. On that prompt two plain ways of cutting the calls differ by 1.25, and their greedy outputs part after the same three tokens as the padded call's.
- Checkpoints taken while decoding, and a request that returns after another one ran in between, stay bit for bit.

While checking this the runner's record of overwritten positions became exact: it assumed a verify's rejected rows after every stretch, also where nothing had been decoded, which made a request that returned after a short one run four positions again through other calls.

## 16. Start-up: programs kept between starts (2026-11-27)

A start of the server compiled for 22 minutes with every kernel's binary and every tuning result already in tinygrad's caches. The programs are now saved after that and loaded at the next start: the server is ready 65 s after it is launched, of which 40 s load the model. Nothing about the programs changes, and the outputs are the same bit for bit. It applies to every model the server runs, not only this one.

### Where a start went

One fixed-length prefill program (768 tokens), the first in a fresh process, timed by phase:

| Call | Seconds | Where |
|---|---|---|
| First: tinygrad runs the graph kernel by kernel | 57–58 | building the graph in Python 9; exporting it 26 (the whole graph 21, and about 5 for the 600 small realizes the model's code makes on its first call); scheduling 8; lowering the kernels in worker processes 8; command buffers 6 |
| Second: captures the program | 59 | exporting the graph again 21; lowering the 105 kernels the JIT tunes, one after another in the main process, about 30; memory plan and command buffers 2; the rest is building the graph |
| Third: runs it | 0.7–0.8 | |

- The compiler does not run in any of this: the time is tinygrad's Python (graph rewriting), paid twice per program.
- The second call took 76 s the first time, when the tuning (BEAM) of 66 of those kernels was not cached yet: 46 s of lowering, 16 of them the search.
- The server builds 31 programs. Seconds for both calls, from its log: the 2,048-token program and the one for up to 256 tokens 256 together, up to 512 tokens 143, up to 2,048 tokens 104, each of the five fixed lengths 110 to 112, the two decode programs 67, draft and verify 72, the copied block's verify 131, the fifteen restore programs 5. In all 1,346 s.

### What was built

`tinyllm.tg.programs` saves, in one file after the warm-up:

- **The captured programs** (tinygrad's `CapturedJit`: compiled calls, not yet linked to addresses).
- **What the first calls set on the model's objects:** the kernel chosen for each layer, weights re-viewed as packed bytes, caches and recurrent state (4,538 attributes). A later start must find the objects as the programs left them, for the code outside the programs (state checkpoints) and for a program built later.

The model is not in the file. Each buffer a program or one of those attributes holds is written in one of three ways (the server's session):

| Buffers | Count | Bytes | Written as | At load |
|---|---|---|---|---|
| Existed when the session was constructed (weights, the MTP block, state the constructor makes) | 1,835 | 87.4 GB | their place among the session's objects | bound to the buffer the new process loaded there |
| Made later and written by a program (attention caches, recurrent and convolution state) | 122 | 1.16 GB | their size | made new and zeroed, as the model makes them |
| Made later, only read (a rotation table, a program's constants) | 3 | 12.1 MB | their contents | copied back |

- **Which buffers a program writes** comes from tinygrad's own lowering: it knows each compiled kernel's output arguments before it merges the calls into command buffers, and tinyllm notes them there.
- **Objects that alias buffers from the host** are left alone: the session itself and the n-gram row reader. The reader's host buffer was made by the first write; it is now made when the session is constructed, so a loaded program and the reader hold the same one.
- **The key.** A file is valid for the model files (names, sizes, modification times), the session's settings, the sources of tinygrad and of `tinyllm/tg`, tinygrad's context variables, the environment variables that reach the kernels (`JITBEAM`, `DEV`, `TINYLLM_*`) and the device. Anything else builds the programs again and saves them beside the old ones; the four most recently used sessions are kept.
- **Where.** `tinyllm serve` keeps them under `~/.cache/tinyllm/programs` (`--program-cache DIR` or `TINYLLM_PROGRAM_CACHE`; `0` compiles at every start). The file is a pickle, so it is only read from a directory and a file that no other user can write, and it may only name tinygrad's and tinyllm's own classes.
- **Refusals.** If something a first call left cannot be written (a function, a large buffer no object holds) nothing is saved and the log says why; if a file does not fit the session it is not loaded. Either way the session builds its programs as before.
- **A load that fails for a passing reason rebuilds.** Seen once (2026-11-28): a start right after other GPU jobs had exited reported `programs not loaded: BlockingIOError: [Errno 11] Resource temporarily unavailable` and compiled everything again; the next start, two minutes later, loaded the same 33 programs in 60 s. `restore` treats every error alike and does not try again. Not fixed yet.

### Start-up times

From process start to ready, seconds:

| Session | Programs built | Programs loaded |
|---|---|---|
| Flash-Next `UD-Q4_K_XL`, the server's settings (2,048-token chunks, 4 MTP drafts, copied blocks) | 1,390 | 65 |
| Flash-Next `UD-Q4_K_XL`, 256-token chunks (the check's) | 644 | 47.6 |
| Qwen3.8-27B `UD-Q4_K_XL`, MTP | 150 | 23.8 |
| Qwen3.8-27B `UD-Q4_K_XL`, DFlash2 | 181 | 29.3 |

- **A loaded start of the server:** 40 s to load the model, 2.1 s to load the 31 programs (50 MB), 17 s to run each once. Saving them took 13 s.
- **The server behaves as before.** Started from loaded programs it passes the SDK's functional checks, and the agent benchmark shows the numbers of the built one: 1.94 to 2.20 s to the first token for steps that add 1,300 to 1,570 tokens, 40 to 56 tok/s decoding.
- **Stopping.** A server that built its programs took 57 s to exit after SIGTERM; one that loaded them takes 3.3 s.

With the padded calls of section 15 the server builds 33 programs: a 512-token prefill program (87 s) and the copy of the state a call keeps were added. That build took 1,451 s, ready after 1,495 s, and the file is 53 MB.

### Correctness

`tools/program_cache_check.py` runs a session twice in separate processes, on an empty program directory. The second process must build no program and produce the same numbers bit for bit: the logits after a prompt, plain decoding, the sampler's results of draft-and-verify cycles, a copied block's verify and the cycles after it, decoding sampled on the GPU, and a second prompt that resumes from a state checkpoint. It passes on Flash-Next `UD-Q4_K_XL` with its MTP block, on the 27B with its MTP block and with DFlash2, and on the small model on the CPU. Tests: `tests/test_programs.py` (the CPU run and the mechanism on a one-layer session in the suite, Flash-Next as a GPU test) and `tests/test_serve_gpu.py`, which starts the server twice and wants the SDK checks to pass with programs built and with programs loaded.

### Faster rebuilds (2026-11-27)

The first start after a change compiled for 1,451 s (33 programs). It now compiles for 867 s, and the programs are the same.

| Build | Seconds |
|---|---|
| Before | 1,451 |
| Tuned kernels lowered in worker processes | 1,187 |
| and programs of a function already captured, captured by their first call | 867 |

- **Tuned kernels in worker processes** (`tinyllm.tg.lowering`). tinygrad lowers a batch of kernels in its pool of worker processes unless one of them is to be tuned by BEAM search, which times its candidates on the device; then it lowers the whole batch in the main process, one kernel after another. A JIT with `JITBEAM` marks every generic kernel that way, so each program's 105 tuned kernels took about 30 s although their searches' results were on disk. A search whose result is on disk never touches the device, so those kernels now go to the pool, and tinygrad lowers in the main process only the ones whose search has to run. The capturing call of a 768-token program went from 59 to 37–39 s.
- **Captured by the first call** (`tinyllm.tg.programs.Program`). tinygrad's first call of a program runs the graph as it is, for what a function does once (weights it realizes, caches it makes); captured, those would run at every call. A program whose function another program has already been through, the same function at another length, has nothing left to do once. Its first call captures it: the eight other prefill programs and fourteen of the fifteen restore programs. A 768-token program is built in 50 s instead of 108, the 512-token one in 37 instead of 87, the one for up to 512 tokens in 65 instead of 140.
- **The check on that.** A capture records one linear per realize. If a first call records another number than the other program's capture did, it made something on the way; that is done by then, and the next call captures again. None did in the server's build. The warm-up now finishes the first program of each function before it calls the other lengths; calling every program once before any twice, as before, left nothing captured in time.
- **Where the 867 s go:** the 2,048-token program and the one for up to 256 tokens 211 (two calls each), the two decode programs 63, draft and verify 56, the copied block's verify 110, the restore programs 3, then one call each: up to 512 tokens 65, up to 2,048 tokens 52, and the six fixed lengths 37 to 54.
- **The same programs.** `tools/program_build_check.py` builds the 768-token program by its first call and again in two calls: logits and state are equal bit for bit. A session with 256-token chunks built this way reproduces every output of the build of that morning (prompts, plain and sampled decoding, cycles, a copied block, a resumed prompt) bit for bit.

Tried, not kept:

- **tinygrad's schedule cache on disk** (`SCACHE=2`). A second process finds the schedules of the first, 6 s less per program whose graph has not changed. Each code version adds about 300 MB to tinygrad's cache file, which nothing evicts. It stays off; setting `SCACHE=2` in the environment turns it on.
- **One realize for all the weights at load** instead of one per tensor: 26 s against 30 in the better of two runs and 60 in the other. The load is the disk's.

### What is left

- **The first start after a change still compiles for about 15 minutes.** Any edit under `tinyllm/tg`, another tinygrad, other settings or another model file makes a new key. What remains per program is single-threaded Python: building the graph 9 s, exporting it 22, scheduling it 9, twice for the first program of each function. Past that lies building the programs in several processes at once.
- **The model's load time varies.** It is 40 s when the previous large job has just released its memory, which is the case of a server stopped and started again. Twice it took 146 and 249 s, both right after the CPU test suite had run; I have not established why (tinygrad reads the file directly, so the suspect is the kernel finding 85 GB of memory for the GPU when free memory is fragmented).
- **Loading the model is most of a start:** about 40 s of the 65 for Flash-Next's 85 GB.
- **`tinyllm bench` and the check tools** build their programs at every run; only the server and `tools/program_cache_check.py` use the program directory.
