# ds4 / DeepSeek V4.1 Flash — decode performance handoff

**Author:** Claude Opus 5 session, 2026-09-14
**For:** Fable 5.1 (or whoever picks this up next)
**Repo:** `/Users/gaineyllc/ds4`, branch `ssd-decode-bank-shrink`, HEAD `b77b624`
**Upstream base:** `a04f46f` (antirez/ds4 `main` at branch point) — **`origin/main` has since moved to `9139e2a`; see §−1 before doing anything**
**Machine:** Neil's MacBook Pro, Apple M5 Max, 128 GiB unified memory, macOS

---

## −1. Before you read anything else: re-sync the repo

**This document describes a branch that was already behind upstream when it was written.** Every measurement, line number, and file reference below is against `ssd-decode-bank-shrink` at `b77b624`. Do not trust a line number in this document until you have diffed.

State at the moment of writing, `2026-09-14 18:08Z` [measured, live `git fetch`]:

| | |
|---|---|
| branch | `ssd-decode-bank-shrink` @ `b77b624`, tree clean |
| branch point | `a04f46f` ("DeepSeek v4.1 Flash support for CUDA") |
| `origin/main` | `9139e2a` ("Document and download self-contained Qwen BF16 n-gram releases") |
| divergence | **33 ahead, 16 behind** |
| local `main` ref | `bd66c40` — **stale, two commits older than the branch point.** Do not rebase onto local `main`; fetch and use `origin/main`. |

**Do this first:**

```sh
cd /Users/gaineyllc/ds4
git fetch origin
git log --oneline a04f46f..origin/main          # what landed upstream
git diff --stat a04f46f..origin/main -- ds4.c ds4_metal.m metal/moe.metal ds4_gpu.h
git rebase origin/main                           # branch is clean; backup-pre-rebase exists
```

### Why this is not a formality

Upstream touched **six of the eight files this branch touches** [measured]:

| file | upstream lines changed since branch point | this branch touches it |
|---|---|---|
| `ds4.c` | +5035 | yes |
| `ds4_metal.m` | +2085 | yes |
| `metal/moe.metal` | +230 | yes |
| `ds4_gpu.h` | +235 | yes |
| `ds4_cuda.cu` | +301 | yes |
| `.gitignore` | +9 | yes |
| `ds4_engram.c`, `gguf-tools/deepseek41_dspark.py` | untouched upstream | yes |

Most of that volume is a new model family (Qwen3.8 Flash Next, `ccea768` / `f06d3eb`) that has no obvious bearing on the V4.1 decode path. **Three upstream commits are in V4.1 territory and you should read them before trusting §3 or §9:**

- `c2c3ce3` — "Overlap CUDA SSD expert reads with V4.1 prefill"
- `e9e1baa` — "Process medium CUDA SSD appends in one layer sweep"
- `6e4c285` — "Record CUDA SSD prefill gains and regression checks"

These are CUDA-side and prefill-side; this branch is Metal-side and decode-side. Whether they overlap conceptually with the deferred-residency design in §4, or invalidate anything in §3, I have not determined — they landed after my last fetch and I have not read them. `8464282` ("Keep short Qwen prefill tails on FP32-query attention") also touches attention paths and is worth a look.

Upstream also added `tests/test_q8_prefill_variants.c` and grew `tests/test_deepseek41_prefill.c` and `tests/ds4_test.c` substantially. Run the full suite after rebasing; the byte-identical-output check in §4 is the one that matters most for the deferred-residency work.

**Conflicts — resolved answer, not an expectation** [measured, `git merge-tree --write-tree HEAD origin/main`, Apple Git 2.50.1 on the Mac]:

Four of the six overlapping files auto-merge clean: `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_metal.m`, `metal/moe.metal`. **Every Metal and kernel change on this branch merges without conflict.** Two files conflict, both trivially:

**1. `ds4.c` — one hunk, ~line 83915.** Upstream and this branch each inserted a speculative-decode dispatch at the same point in the same function. They are mutually exclusive model kinds, so the resolution is to keep both:

```c
#ifdef DS4_HAS_DEEPSEEK41_GPU
    if (ds4_session_is_ds41(s) && s->engine->support_kind == DS4_SUPPORT_DSPARK &&
        s->engine->dspark && !s->engine->quality && !s->engine->dspark_strict &&
        accepted && accepted_cap > 0) {
        return ds4_session_ds41_dspark_cycle(s, first_token, max_tokens, eos_token,
                                             accepted, accepted_cap, err, errlen);
    }
#endif
    if (ds4_session_is_qwen4(s)) {
        (void)max_tokens;
        (void)eos_token;
        if (!accepted || accepted_cap <= 0) return 0;
#ifdef DS4_HAS_QWEN4_METAL
        if (s->engine->glm_mtp && DS4_N_NEXTN_PREDICT != 0 && s->qwen4_graph_ready) {
            return ds4_session_qwen4_spec_cycle(s, first_token, 0.0f, 0, 0.0f, 0.0f, NULL, false,
                                                accepted, accepted_cap, err, errlen);
        }
#endif
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
```

I have not verified that this ordering is correct beyond the fact that the two guards cannot both be true (`ds4_session_is_ds41` vs `ds4_session_is_qwen4`). [unverified] whether any later code in that function assumes one ran.

**2. `.gitignore` — one hunk.** Keep all three lines: `hf-dspark/` (mine), `/tests/test_qwen4_conv_parallel` and `/tests/test_q8_prefill_variants` (upstream).

`backup-pre-rebase` exists as a branch if you need to get back.

**After rebasing, re-measure before trusting any number in §0.** The throughput figures are the single most rebase-fragile thing in this document.

---

## How to read this document

Every performance number below carries its **provenance**: how many runs, warm or cold, and whether anything else was competing for the machine. That matters more than usual here — see §6, this machine actively fights benchmarking.

**Nothing in this document predicts the outcome of work that has not been done.** Where a direction exists, §9 states the measured facts that motivate it and the code that would have to change. It does not estimate what throughput would result. Those determinations are yours.

Claims are labelled:
- **[measured]** — observed, with run count.
- **[derived]** — arithmetic from measured or file-layout facts.
- **[unverified]** — a hypothesis I did not test. Treat as a lead, not a finding.

---

## 0. State

Neil's goal: DeepSeek V4.1 Flash Q2 as a final reviewer of large codebases, **1M token context (non-negotiable)**, **≥50 tokens/second generation**, fast prefill, concurrent request serving.

**Current generation throughput** — all on branch HEAD `b77b624`, `DS4_METAL_STREAM_EXPERT_NOCOPY=1`, nothing else running:

| condition | t/s | provenance |
|---|---|---|
| ctx 32k allocated, ~10-token prompt, warm | **20.9 / 22.4 / 22.5** | [measured] 3 runs, run 1 of a set discarded as cold |
| ctx 1M allocated, ~10-token prompt, warm | **19.5 / 20.2** | [measured] 2 runs |
| ctx 1M allocated, ~10-token prompt, cold | 8.7 | [measured] 1 run, first after model load |
| ctx 131k allocated, **65k-token prompt** | **9.5 – 10.2** | [measured] single runs per arm, server verified down |
| stopping path (deferral off), ctx 32k, warm | 18.3 / 18.4 / 18.5 | [measured] 3 alternating pairs vs the above |

For reference, the prior session's starting point on this workload was 16.0–18.5 t/s.

**A number I do not have:** nothing here was measured with a genuinely full 1M-token context. Every "1M" row above is an *allocated* window with a ten-token prompt. The largest real occupancy tested is 65k. See §9 Lead 3.

### The central measured fact

Decode reads **9.79 GB of weights per generated token** [derived, §3 — from GGUF file layout, not from assumed type sizes].

At the measured 22 t/s that is **215 GB/s**. Against a **~546 GB/s** figure for M5 Max peak memory bandwidth, that is 39.5% of peak, and 50 t/s would require 490 GB/s — 89.7% of peak.

⚠️ **The 546 GB/s figure is from my training data and was never measured on this machine.** Every conclusion about headroom rests on it. Establish the *achievable* streaming bandwidth first — a Metal kernel doing a large strided read with no compute — before using §3 for anything. If the achievable figure differs materially from what I assumed, the whole picture changes.

Four optimization attempts (§5) failed, and all four are consistent with bandwidth being the binding constraint. That is a pattern, not a proof.

---

## 1. Hardware, model, file inventory

**Machine:** Apple M5 Max, 128 GiB unified memory. Metal 4 runtime, MTL4 queue, tensor API, "M5 neural accelerators likely" per ds4's own startup probe.

**Model files**, `/Users/gaineyllc/ds4/gguf/`:

| file | size | notes |
|---|---|---|
| `DeepSeek-V4.1-Flash-Q2.gguf` | 365,713,686,528 B (340 GiB) | target. IQ2_XXS gate/up, Q2_K down, Q8_0 attention/shared/head |
| `DeepSeek-V4.1-Flash-DSpark-native.gguf` | 8,421,605,376 B (7.84 GiB) | DSpark drafter, near-native precision (§7) |
| `DeepSeek-V4.1-Flash-DSpark-support.gguf` | 4,568,907,776 B (4.25 GiB) | DSpark at Q2 — **does not run** (§7.3) |
| `DeepSeek-V4.1-Flash-Vision.gguf` | 970,555,552 B | vision tower — see below |

**On the vision tower:** it was exercised in exactly one way — a startup run with `--vision` to determine whether it enters the expert-cache budget. **It does not** [measured, 1 run]: the cache target came out at exactly 65.63 GiB and static weights locked at 9.37 GiB, byte-identical to a run without `--vision`. Vision *inference* (image input) was never exercised, and no vision code path was touched on this branch. The budget result is in §2; it is the same gap that affects the support model.

**HF source shards** (`/Users/gaineyllc/ds4/hf-dspark/`, gitignored): shards 44, 45, 46 of 48, plus `config.json` and `model.safetensors.index.json`. These are the three containing `mtp.*`. **Shards 47–48 were never downloaded** — they hold the Engram tables, so re-deriving Engram from source requires fetching them.

**Architecture facts** (`hf-dspark/config.json`, `text_config`):

```
hidden_size 5120, moe_intermediate_size 2304, num_hidden_layers 40
num_attention_heads 64, num_key_value_heads 1, head_dim 512
q_lora_rank 1280, o_lora_rank 1024, o_groups 8
n_routed_experts 384, n_shared_experts 1, num_experts_per_tok 6
sliding_window 128, max_position_embeddings 1048576
rope yarn factor 16, original_max_position_embeddings 65536
engram_layer_ids [1, 14], engram_max_ngram_size 4
engram_num_embeddings [384006168, 384016682], engram_vocab_size 16000000
engram_n_heads 8, engram_head_dim 256, engram_compressed_vocab_size 99092
dspark_block_size 5, dspark_markov_rank 256, dspark_target_layer_ids [37,38,39]
dspark_n_routed_experts 128, dspark_num_experts_per_tok 3
dspark_noise_token_id 128799
quantization_config: fp8, weight_block_size [32,32], scale_fmt ue8m0, expert_dtype fp4
```

**Where the 340 GiB goes** [derived from GGUF tensor table]:

| component | size | share |
|---|---|---|
| Engram n-gram tables (layers 1 and 14) | **188.8 GiB** | 56% |
| routed experts (40 × 384 × 9.4 MB) | 142.3 GiB | 42% |
| static / attention weights | 9.37 GiB | 3% |

Each Engram table is `[264, 384006168]` — 384 million rows of 264 bytes = 94.4 GiB, twice. GGUF tensor type 24, encoding `deepseek41.engram.encoding = e4m3_e8m0_32_row264`. ds4 logs `Engram disk-only` at startup and reads them with `pread` via `ds4_engram_table_open` / `ds4_engram_read` — not through the expert cache, not through any GPU mapping.

I reasoned about the expert cache as a fraction of "the 340 GiB model" for most of the session. That was wrong: the expert working set is 142 GiB, so the 65.63 GiB decode bank at 1M covers **46% of the experts**, not 19%.

---

## 2. The expert cache

`--ssd-streaming` keeps a bank of routed experts resident and streams the rest.

- Auto-sized at `ds4.c` ~64470 (`ds4_ssd_auto_cache_plan`), fed `recommended` (Metal `recommendedMaxWorkingSetSize`), `cache_percent` 86 for non-GLM, `model_limit` from `ds4_streaming_manual_cache_safe_bytes`, and `non_routed_bytes` from `weights_streaming_non_routed_bytes(&e->weights, …)`.
- Budget observed [measured]: **76.62 GiB** at ctx 32768, **75.62 GiB** at ctx 131072, **65.63 GiB** at ctx 1048576.
- Per-expert 9.49 MiB (gate + up + down).
- `--ssd-streaming-cache-experts 95GB` is clamped to 75.00 GiB by `ds4_streaming_manual_cache_safe_bytes` [measured].

**Neither auxiliary model enters the budget** [measured, 4 runs]. The cache target is byte-identical with and without `--vision`, and with and without `--mtp-model`. `weights_streaming_non_routed_bytes` reads only the target model's static spans. On this machine it caused no observed memory pressure: at 1M with DSpark loaded, mid-decode sampling showed `free=55.9 wired=4.6 compressed=4.5 GiB` and **zero swapouts**; without DSpark, `free=50.3 wired=4.6 compressed=4.5`, also zero.

**Two cache modes:**

- **Copy mode** (default) — experts `pread` into wired slab buffers.
- **No-copy mode** (`DS4_METAL_STREAM_EXPERT_NOCOPY=1`, added on this branch) — experts stay in the model file's own mmap'd pages, wrapped with `newBufferWithBytesNoCopy`. `bind_total` 2242 → 96 ms, `cache_mixed` 2632 → 0, hit_rate 0.765 → 0.820 [measured].

**The two modes invert with context length** [measured, single runs per arm, server verified down]: at 65k occupied context, copy mode 10.21 t/s vs no-copy 4.94. At short context the prior session measured no-copy +24% at 8k. There is no adaptive switch — it is a manual env var with no in-code guidance.

---

## 3. Weight traffic per token

Method: read the GGUF tensor table; compute bytes-per-weight from **consecutive tensor data offsets in the file** rather than assumed type sizes; scale `*_exps` by 6/384 since top-6 of 384 are read per token.

I derived this three times. The first two were wrong — once by counting only routed experts at a wrong per-expert size (this produced a "9% of bandwidth" claim I stated to Neil and later retracted), once with guessed type codes. **Re-derive it rather than inheriting it.**

Bytes/weight from layout: `F32 4.0, F16 2.0, Q8_0 1.0625, Q2_K 0.3281, IQ2_XXS 0.2578`.

| tensor (all 40 layers) | GB/token | share |
|---|---|---|
| `attn_q_b.weight` | 1.78 | 18.2% |
| `attn_output_b.weight` | 1.78 | 18.2% |
| `attn_output_a.weight` | 1.43 | 14.6% |
| `ffn_down_exps.weight` | 0.93 | 9.5% |
| `ffn_gate_exps.weight` | 0.73 | 7.5% |
| `ffn_up_exps.weight` | 0.73 | 7.5% |
| `ffn_gate_shexp.weight` | 0.50 | 5.1% |
| `ffn_up_shexp.weight` | 0.50 | 5.1% |
| `ffn_down_shexp.weight` | 0.50 | 5.1% |
| **TOTAL** | **9.79** | |

Required bandwidth [derived], against the **unverified** 546 GB/s peak:

| rate | GB/s | % of peak |
|---|---|---|
| 11 t/s | 108 | 19.7% |
| 22 t/s (current) | 215 | 39.5% |
| 50 t/s (goal) | 490 | 89.7% |

Arithmetic ceiling at 100% of that peak: **55.8 t/s**.

**Attention is 51% of weight traffic** (`q_b` + `output_a` + `output_b` = 4.99 GB/token), all Q8_0. Routed experts are 24.5%. `head_dim` 512 × 64 heads makes `attn_q_b` alone `[1280, 32768]` = 42M params = 44.6 MB per layer at Q8_0.

---

## 4. What shipped

19 commits on `ssd-decode-bank-shrink`. DSpark conversion/binder (`f33db6a` … `7734064`) predates this session; decode performance (`65b7de8` … `b77b624`) is this session and the prior one. Diffstat vs `d32661a`: `ds4.c` +1316, `ds4_metal.m` +889, `gguf-tools/deepseek41_dspark.py` +431 (new), `ds4_gpu.h` +22, `ds4_cuda.cu` +15, `metal/moe.metal` +14.

Build is clean — **zero compiler warnings** (antirez's new `QA_BEFORE_RELEASES.md` treats warnings as build failures).

### 4.1 `65b7de8` — async per-layer commit

`ds4_gpu_end_commands_async()` commits and appends to `g_pending_cbs`, capped at `DS4_GPU_MAX_ASYNC_BATCHES` (8). Blocking commits retained only at layer 13 (Engram input hazard) and layer 39 (publishing the token). **+26% at 8k** [measured, prior session].

`ds4.c` ~41095:
```c
const bool overlap = queue_layers && g->tp_world != 2 &&
    !getenv("DS4_METAL_DISABLE_V41_DECODE_OVERLAP");
const bool drain = !queue_layers || overlap || il == 13 || il + 1u == DS4_N_LAYER;
const bool blocking = !overlap || il == 13 || il + 1u == DS4_N_LAYER;
```

### 4.2 `57adcc4` / `0aa8f4a` — no-copy expert cache

**+24% at 8k, +20% at 1M, byte-identical output** [measured, prior session]. `0aa8f4a` disabled it when a support model was loaded; `e4b81b1` later fixed the underlying cause and re-enabled.

### 4.3 `c464574` — deferred expert residency

**Observation** [measured]: the routed MoE stopped 40× per token to read the router's choice off the GPU, check those six experts were cached, and hand their addresses to the matmul. That stop was almost entirely *waiting*: `sync_avg` 1.471 ms vs `bind_avg` 0.016 ms. And across a full decode run **every layer found all six experts already resident** — `cache_all_resident=4320, cache_all_missing=0, cache_mixed=0`.

**Change:** the address-table kernels (`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32`, `kernel_mul_mv_addr_q2_K_sum6_f32`) read the router's own output buffer and index a per-layer address table that install/evict already keep current. They return without contributing when an address is zero — a miss costs a missing contribution, not a fault. Residency is checked on the GPU, the token runs straight through, the result is read once at the end. If any layer came up short the token is rolled back and re-decoded with stops in place.

Three supporting changes were required:

1. **Residency.** Raw GPU addresses are invisible to Metal, so each dispatch used to name the six buffers with `useResource:` — only possible if the host knows which six. The bank now lives in one `MTLResidencySet` handed to the queue (`ds4_metal.m` ~15157). Membership changes only on install/evict.
2. **Eviction interlock.** A deferred token cannot pin entries inflight because it never named them, so nothing is evicted while one is in flight (`ds4_gpu_stream_expert_defer_in_flight()`).
3. **The rollback snapshot** stopped borrowing DSpark's carry buffers, which only exist when a DSpark module is loaded. **This is why an earlier version silently disabled itself on every token.** It now has its own `defer_carry_kv` / `defer_carry_score`.

**Result** [measured]: 8120 of 8120 layers deferred, zero redos, output byte-identical over 150 tokens on one prompt. **21.09 / 21.14 t/s vs 18.44 / 18.48** on the stopping path — two alternating pairs, warm, plus an earlier set of three alternating pairs at 20.93/20.87/20.74 vs 18.34/18.30/18.30. This is the best-evidenced result on the branch.

**Invariant to preserve:** a token commits to one mode at its first routed layer (`g_stream_expert_defer_token_mode`) so a stopping layer — which must be free to evict — can never sit inside a deferred layer's window.

### 4.4 `e4b81b1` — no-copy cache serves DSpark

Two of four expert loaders never learned about no-copy: `ds4_gpu_stream_expert_cache_load_selected_missing_with_source` (~17604) and `ds4_gpu_stream_expert_cache_prepare_selected_batch` (~18556) still allocated a slab slot and `pread` into it. Under no-copy those "slots" are views of a **read-only mapping**, so the read failed with **EFAULT**, returning zero bytes. That was the entire content of `V4.1 DSpark verify failed` — DSpark's verify batch is the one decode path through those loaders. Both now wrap.

Also fixed here: both cache prune loops spun forever when `clear_entry` declined an eviction (they loop until the entry count drops); and the residency set treated additions and removals alike, so pruning marked it dirty, a dirty set turned deferral off, and the stopping path pruned again — a livelock between two features. Only additions need a commit before use; removals need the buffer to outlive the set's reference, so retired buffers are held until the next commit.

### 4.5 `7a4fcb5` — rollback snapshot rides the token's command buffer

The snapshot was its own commit-and-wait at the head of every token — a full GPU drain. Command buffers on one queue run in order, so the copies only need to be *encoded* ahead of the token's work. `ds41_verify_snapshot_encode` (`ds4.c` ~40922). No throughput change measurable in isolation once the other changes were in.

### 4.6 `34e758f` — kernels self-report misses; exponential backoff

**Wrong place to ask:** a 1×1×1 validator dispatch per layer sits in the compute encoder as a barrier between the MoE and everything after it. Removing it moved 65k decode from 6.94 → 9.94 t/s [measured, 1 run each]. The address kernels already discover a miss, so each now raises one relaxed atomic on that path (`metal/moe.metal`, `atomic_fetch_or_explicit(miss, 1u, memory_order_relaxed)` at the two `addr == 0` early-outs) and the dispatch is gone.

**Wrong question:** the old validator was *also reporting no misses when there were misses*. With honest reporting, 65k context misses on **every token — 39 redos in 39 tokens** [measured]. Once the KV cache is large enough the expert bank no longer holds a token's working set, so the deferral bet cannot be won and every deferred token is decoded twice.

So the bet is abandoned when it starts losing: exponential backoff 8 → 4096 tokens with an occasional probe, reset after 256 clean tokens (`DS4_METAL_DEFER_BACKOFF_MIN/MAX/FORGIVE`, `ds4_metal.m` ~15646).

At 65k [measured, 1 run per arm]: **9.49 t/s with backoff, 3 tokens deferred of 40**, against 10.23 for the pure stopping path and 6.94 for unbacked-off deferral. At short context: 8120/8120 layers deferred, **redos=0, backed_off=0**.

### 4.7 `276a3fe` — support model weights held resident

`vmmap -resident` mid-decode showed the DSpark mapping at **3.0 MiB resident of 7.84 GiB**, while the target held its 9.4 GiB of `mlock`'d statics. Only `e->model.map` was ever locked; the drafter is read through no-copy views of its own mapping, so every draft step faulted its weights off SSD. Now `mlock`'d like the target's statics (`DS4_DISABLE_SUPPORT_MODEL_MLOCK` opts out): **7.84 GiB locked, 0.00 pageable**.

Throughput after: **6.13 t/s** [measured, 1 run] against 5.75 / 5.84 / 5.90 before [3 runs]. **Thin evidence — one post-change run.** The residency fact is solid; the throughput delta is not well established.

### 4.8 `54b9383` / `b77b624` — Engram decode cost, measured and partly hidden

`DS4_DECODE_PROFILE_DETAIL` only instruments the *prefill* sweep; decode-side Engram reads were uncovered. Added `DS4_ENGRAM_DECODE_PROFILE`. Measured **5.493 ms/token** across 212 reads at ctx 131072 — 24 scattered 264-byte rows per table out of a 94 GiB file, synchronous at the head of every token.

Layer 1 needs `rows[0]` almost immediately; **layer 14 does not need `rows[1]` until thirteen layers have gone by**. The second read now runs on its own thread and joins just before layer 14 binds it, with a second join after the layer loop so the thread cannot outlive the frame whose ids and destination it uses (`ds41_engram_decode_bg`, `ds4.c` ~41068).

[measured, 2 alternating pairs, ctx 131072, short prompt]:

```
serial   3.947 / 3.857 ms/token engram     12.58 / 12.61 t/s
overlap  2.059 / 2.090 ms/token engram     12.93 / 12.91 t/s
```

Residual at the join ~0.08 ms — the layers cover it. Byte-identical over 150 tokens.

⚠️ **Provenance caveat:** absolute throughput in that pair (12.6–12.9) is well below comparable short-prompt runs measured later with the server verified down (20–22 t/s). Those runs predate my discovery of the `ds4-server` contention problem (§6.1) and were very likely contended. The **delta** comes from alternating back-to-back runs and the engram-ms figures are direct, so the improvement is credible; the **absolute numbers in that block are not** comparable to the §0 table.

The remaining ~2 ms is the layer-1 table, whose ids depend on the token just sampled.

---

## 5. What was tried and failed

All measured. Do not repeat without new information.

| attempt | result | provenance |
|---|---|---|
| **Vectorized Q8_0 matvec.** `block_q8_0` is 34 bytes so `qs` is never 4-byte aligned — hence the shipped kernel's 8 scalar `int8_t` loads. Metal `packed_char4` has alignment 1; wrote `kernel_mul_mv_q8_0_f32_4` with `packed_char4` + `packed_float4`, bit-identical arithmetic, 4× fewer load instructions. | **20.55 / 20.38 vs scalar 21.79 / 20.67 — no gain.** Reverted. | 3 alternating pairs, first discarded as cold |
| **`DS4_METAL_Q8_MV_NSG` sweep** (2 / 4 / 8) | 9.82 / 16.15 / 12.62 — 4 is best of the three | **1 run each, cold.** Ordering is stark but evidence is thin |
| **Fusing BF16 rounding into producing kernels.** `kernel_dsv41_bf16_linear` is a pure in-place BF16 round, 9.7 dispatches/layer at 8.1 µs each for ~20 KB of work. Fused into `kernel_dsv4_hc_expand4`'s store (2 of ~9.7 sites/layer). | 21.21 / 22.03 — within noise of baseline. Reverted. | 3 runs |
| **Concurrency** — server, 1 / 2 / 4 simultaneous completions | aggregate **11.0 / 11.2 / 11.3 t/s**, per-stream exactly 1/N | 2 rounds; cold round gave 6.1 / 10.0 / 10.9 |
| **Bigger expert cache at 1M** (`--ssd-streaming-cache-experts 95GB`, 65.63 → 74.99 GiB) | 12.81 → 12.72 | 1 run each |
| **Dropping `requestResidency`** on the cache set | no change (8.60 → 8.73) | 1 run each |
| **`DS4_METAL_MAX_ASYNC_BATCHES=1`** | 6.33 vs 6.94 | 1 run each |
| **One command buffer per token** (`DS4_METAL_DISABLE_V41_DECODE_OVERLAP=1`) | 20.11 / 20.08 vs 20.74 / 20.74 | 2 alternating pairs |
| **Replaying deferred routing to the cache** (hotness + prefetcher + pruners) | neutral; pruning measured worse (3.34 vs 3.83 at 65k) | 1 run each; shipped without pruners |

### ⚠️ A false result I nearly built on

Skipping the BF16 rounding entirely measured **+24%** (15.20 → 18.59, 16.41 → 20.75) and looked like the largest prize of the session.

It was an artifact. Rounding is idempotent, so I ran it **twice** — numerics-preserving, one extra dispatch per site — and it **cost nothing** (21.64 / 21.64 vs 19.89 / 21.94). The extra dispatch is free; the +24% came from *different rounding → different tokens → different expert routing → different cache behaviour*.

**Generalize this:** on this system you cannot A/B a flag that changes numerics and attribute the delta to performance. Always construct a numerics-preserving version of the experiment — do the work twice, not zero times.

---

## 6. Measurement methodology and traps

Read this before measuring anything.

### 6.1 `ds4-server` will corrupt your measurements

`~/Library/LaunchAgents/com.ds4.server.plist` runs `~/.ds4/start-server.sh`, starting `ds4-server` with `--ctx 1048576`. It **respawned at least six times** during this session despite `launchctl bootout` **and** `launchctl disable`, sometimes within seconds. Two failure modes:

- it takes the **single-instance lock**, so your run dies with `another ds4 process is already running (pid NNNN)` — into a log you may then misread as a result;
- it holds a 1M-context working set, wrecking memory-dependent measurements. A batch of long-context numbers I collected mid-session ranged 3.5–15.8 t/s for identical configurations because of this.

Before every run:
```bash
launchctl bootout gui/$UID/com.ds4.server 2>/dev/null
launchctl disable gui/$UID/com.ds4.server 2>/dev/null
pkill -9 -f ds4-server; sleep 6
ps -eo command | grep -c "[d]s4-server"     # verify 0 — beware greps matching themselves
```
After the run, check the log for `already running` before believing the number.

It is currently booted out and disabled. To restore: `launchctl enable gui/$UID/com.ds4.server`.

(antirez's `QA_BEFORE_RELEASES.md` says *"Do not run multiple huge model processes at the same time."*)

### 6.2 Cold vs warm is a 2× effect

The first run of any set is cold and meaningless — 8.67 vs 20.21 at 1M for the identical command. Discard run 1. Run A and B arms **alternating and back-to-back**, never A×3 then B×3. A trustworthy result looks like `defer 20.93 / 20.87 / 20.74` vs `sync 18.34 / 18.30 / 18.30`. Scatter means something else is running.

### 6.3 Background scripts get killed

Multi-iteration benchmark scripts launched with `nohup … &` through the Desktop Commander bridge were repeatedly killed when the invoking shell exited — several died after one iteration. Run long measurements one invocation at a time and poll, or verify the script is still alive.

### 6.4 Memory: sample mid-run

`vm_stat` "Pages free" measured just after a process exits is reclaim lag, not footprint — it produced a spurious 39 GiB difference that I initially reported as real. Sample during steady-state decode.

### 6.5 Instrumentation available

| env | gives |
|---|---|
| `DS4_METAL_ENCODER_TIMELINE=/path.csv` | per-encoder GPU timestamps via counter sampling. Format: `E <seq> <idx> <start_ns> <end_ns> <dur_us> <gap_us> <caller_unslid> <n_dispatch> <tg> <tpt> <kernel>`; `B <seq> <n_encoders> <gpu_start_ns> <gpu_end_ns>` per command buffer. **Kernel name is the last field.** Forces one encoder per dispatch, inflating wall time — use for relative kernel breakdown, not absolute occupancy. Caller address is inside `ds4_gpu_dsv41_quantize` for all rounding passes, so it does not separate call sites. |
| `DS4_METAL_GPU_BUSY_PROFILE=1` | cumulative GPU busy ms and command-buffer count |
| `DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1` + `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` | cache hit/miss, per-layer stats, `sync_avg`/`bind_avg`/`copy_avg`, and `deferred expert residency: layers=N tokens=N redos=N backed_off=N` |
| `DS4_ENGRAM_DECODE_PROFILE=1` | decode-side Engram read time per token |
| `DS4_V41_DSPARK_LOG=1` | per-cycle `n=`, `agreed=`, `committed=`, `verify=…ms`, `draft=…ms` |
| `DS4_METAL_V41_DEFER_DEBUG=1` | which gate condition blocked deferral, per layer |
| `DS4_METAL_MODEL_VIEW_DEBUG=1` | model-view table dump on a coverage failure |

### 6.6 GPU occupancy picture

`DS4_METAL_GPU_BUSY_PROFILE`: 3931 ms busy over 12,672 command buffers. Timeline restricted to the decode tail (last 160 command buffers = 4 tokens × 40 layers): **1,388 dispatches/token, 34.7/layer**; GPU busy 25.1 ms in an 85.2 ms instrumented token.

Per-token kernel time, decode tail [measured, instrumented — absolute values inflated, relative shares valid]:

| kernel | ms/token | n/layer | µs each | share |
|---|---|---|---|---|
| `kernel_mul_mv_q8_0_f32` | 11.25 | 3.5 | 81.1 | **35.1%** |
| `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` | 3.77 | 0.5 | 191.0 | 11.8% |
| `kernel_dsv41_bf16_linear` | 3.14 | 9.7 | 8.1 | 9.8% |
| `kernel_mul_mv_addr_q2_K_sum6_f32` | 2.52 | 0.5 | 127.6 | 7.9% |
| `kernel_swiglu_flat_f32` | 1.74 | 0.5 | 88.3 | 5.4% |
| `blit:tensor_copy` | 1.74 | 1.1 | 37.9 | 5.4% |
| `kernel_dsv4_attn_out_low_q8_0_f32` | 1.56 | 0.5 | 78.9 | 4.9% |

`kernel_mul_mv_q8_0_f32` at a third of kernel time corresponds to the attention projections that are 51% of the bytes in §3.

---

## 7. DSpark

### 7.1 Precision — mostly native, six tensors are not

The HF release stores DSpark's experts as 1,152 `I8` tensors (packed fp4 pairs) + 1,177 `F8_E8M0` scales — **exactly MXFP4's structure**. `gguf-tools/deepseek41_dspark.py` repacks them **bit-exact** into 9 stacked tensors (`mtp.{0,1,2}.ffn_{gate,up,down}_exps`, `[5120, 2304, 128]`, MXFP4). Nibble order differs (source packs `(2i, 2i+1)` per byte; MXFP4 packs `(j, j+16)`); values identical. The 2,401 → 81 tensor-count drop is **stacking, not dropping** — verified against the GGUF tensor table.

**Six tensors are genuinely requantized:** `mtp.{0,1,2}.attn_output_{a,b}` are **Q8_0**, from the release's `F8_E4M3` + E8M0. Forced — `ds4_gpu_dsv41_attention_output_batch` (`ds4_metal.m` ~27131) hardcodes `ds4_gpu_attention_output_q8_batch_impl` with no type parameter.

Quantization error, measured by dequantizing the released fp8 blocks and round-tripping:

```
mtp.0.attn.wo_a   Q8_0 rel-err 5.46e-03    F16 rel-err 0.00e+00
mtp.0.attn.wo_b   Q8_0 rel-err 5.42e-03    F16 rel-err 0.00e+00
```

F16 is exactly lossless (fp8 fits F16). Whether 0.5% on two matrices affects acceptance is **[unverified]** — I did not test a F16 variant, which would need a new kernel.

### 7.2 Economics [measured]

`DS4_V41_DSPARK_LOG=1`, 21 cycles at ctx 131072:

```
n=4.90 rows    agreed=1.62    committed=1.10
verify=254.8 ms    draft=0.0 ms
5 of 21 cycles committed nothing (parity fallback)
```

255 ms ÷ 1.10 committed = **232 ms/token** against 76 ms for plain decode in the same conditions.

**52 ms per verify row against 76 ms per full token.** The batch barely amortizes, because on a sparse MoE each speculative row routes independently and drags its own 6 experts per layer. Confirmed by counting: **1,021 expert lookups per output token with DSpark vs 415 without** (61,294 vs 24,897 for the same 60 tokens).

[derived] At perfect 5-of-5 acceptance the cycle would be 255/5 = 51 ms/token vs 76 — a 1.5× gain. That is the arithmetic ceiling of the feature *as currently structured*, on the measured verify cost.

Note also: the deferral work in §4.3 removed the fixed per-token overhead that speculation was positioned to exploit.

### 7.3 Known DSpark defects

- **The Q2 drafter does not run.** `DeepSeek-V4.1-Flash-DSpark-support.gguf` fails with `V4.1 DSpark draft failed at position 0`. **Not diagnosed.** One unverified lead: `use_iq2_selected_slots` requires `n_expert == 6`, and DSpark is 128 experts top-3, so IQ2_XXS/Q2_K at `n_expert == 3` may fall to a branch without support. **[unverified]** — I did not instrument the failure.
- Because of that, the hypothesis that low acceptance stems from **drafter/target quantization mismatch** (a near-exact drafter predicting what the unquantized model would say, against a Q2 target) is **[unverified]** — it could not be tested without a working Q2 drafter.
- **The parity constraint discards 24% of cycles** — 5 of 21 committed nothing after paying the full 255 ms verify. The compressor's ratio-2 pooled carry forces commits to an even `(start + commit)`; when that cannot be satisfied the cycle is discarded.
- Acceptance measured 1.62/5 here vs ~2.02 in an earlier session — it drifts.

---

## 8. Upstream state (antirez), 2026-09-14

Three commits on `origin/main`, all **CUDA + prefill**:

```
6e4c285  Record CUDA SSD prefill gains and regression checks
e9e1baa  Process medium CUDA SSD appends in one layer sweep
c2c3ce3  Overlap CUDA SSD expert reads with V4.1 prefill
```

**Zero Metal files touched.** Every `ds4.c` hunk is inside `#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)`; `ds4_gpu.h` additions are CUDA-guarded. **No conflict with this branch.** Diffstat: `ds4_cuda.cu` +301, `ds4.c` +27, `ds4_gpu.h` +9, tests +99, `QA_BEFORE_RELEASES.md` +86 (new), `docs/DGX_SPARK.md`.

Factual notes relevant to a future PR:

- `c2c3ce3` implements read-ahead of the next layer's experts into reserved cache slots, overlapped with compute, for **prefill** — two 8 MiB staging buffers, enabled from 2K tokens.
- `6e4c285`'s message states: *"Include the small automatic-cache decode cost instead of claiming a decoding speedup."*
- `QA_BEFORE_RELEASES.md` names `mac-m5max-it` and `mac-m5max-us` as the preferred Metal/distributed QA hosts — M5 Max, the same silicon as Neil's — mandates zero compiler warnings, and forbids concurrent huge-model processes.

---

## 9. Directions, with the facts that motivate them

**No outcome estimates here.** Each entry gives the measured motivation and the code that would change.

### Lead 1 — attention projections at Q4_K instead of Q8_0

**Motivation** [derived, §3]: `attn_q_b` + `attn_output_a` + `attn_output_b` = 4.99 GB/token = 51% of weight traffic, all Q8_0 at 1.0625 B/weight. Q4_K is 0.5625 B/weight.

**What would have to change:**
- Requantize the target. `gguf-tools/deepseek41_quantize.py`'s `q2` profile sets `att=QTYPE_Q8_0`. This is a 340 GiB rebuild; the ds4 volume had 191 GiB free at last check, so space needs arranging.
- A **decode-path** Q4_K attention-output kernel. `ds4_gpu_attention_output_q4_K_batch_tensor` exists (`ds4_metal.m` ~27154) but early-returns on `n_tokens < 32u` — it is prefill-only.
- Relax `tensor_expect_layout(l->attn_output_a, DS4_TENSOR_Q8_0, …)` at `ds4.c:5641` and the sibling checks at 5539/5541.
- Output changes. Validation on real review tasks would be needed; perplexity alone would not establish fitness for Neil's use.

### Lead 2 — two machines, tensor parallel over Thunderbolt 5

**Motivation:** Neil has a second M5 Max. `tp_world == 2` support is threaded through `ds4.c` / `ds4_metal.m`, and antirez QA-tests this configuration on `mac-m5max-it` / `mac-m5max-us` over TB5 (his QA doc notes TB5 is preferred but fragile and sometimes requires `ds4` in the foreground).

**Known interaction:** the deferral work in §4.3 is disabled under TP — `g->tp_world != 2` in the `overlap` gate, and `tp_world == 2` short-circuits the defer gate. Whether deferral can be made TP-safe, or whether TP needs it, is undetermined.

### Lead 3 — measure a genuinely full 1M context

**Motivation:** this number does not exist. Everything at "1M" in this document is an allocated window with a ten-token prompt; largest real occupancy tested is 65k. Measured points: ~10 tokens → 20.2 t/s; 65k → 9.5–10.2 t/s. The shape of the curve beyond 65k is unknown. V4.1 has `sliding_window` 128 plus compressed KV, so whether it continues linearly is an open question.

**What it takes:** a ~1M-token prompt. Prefill on the 65k prompt measured ~700 t/s; extrapolating that rate to 1M tokens is itself an assumption.

### Lead 4 — determine why concurrency is flat

**Motivation** [measured]: aggregate throughput 11.0 / 11.2 / 11.3 t/s at concurrency 1 / 2 / 4, per-stream exactly 1/N.

**What to establish:** whether `ds4-server` time-slices slots or batches them into shared forward passes. `server_slot` / `slot_threads` in `ds4_server.c` (~9464, ~9591). The two cases have different implications and I did not distinguish them.

Reproduce with the server on a spare port:
```bash
./ds4-server -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --metal \
  --ctx 32768 --host 127.0.0.1 --port 8111
# then N concurrent POSTs to /v1/completions, measure wall time and summed usage.completion_tokens
```

### Lead 5 — no-copy mode has no adaptive switch

**Motivation** [measured]: no-copy is +24% at 8k and 4.94 vs 10.21 t/s at 65k. It is selected by a manual env var with no in-code guidance and no context-dependent default.

### Lead 6 — auxiliary models are outside the cache budget

**Motivation** [measured]: `ds4_ssd_auto_cache_plan` ignores both the vision tower and the support model; the cache target is byte-identical with and without each. No memory pressure was observed on this 128 GiB machine (§2).

### Directions where I have contrary evidence

Not prohibitions — the evidence and its weight, for you to weigh:

- **DSpark acceptance tuning.** The measured verify cost caps the feature at 1.5× even at perfect acceptance (§7.2). Changing the verify structure would change that ceiling.
- **Dispatch-count reduction / kernel fusion.** Extra dispatches measured free (§5, the "twice" experiment, 2 pairs).
- **Q8_0 matvec micro-optimization.** One vectorization approach (`packed_char4`) measured slightly slower [3 pairs]. That rules out that approach, not all approaches.
- **Larger expert cache.** 65.63 → 74.99 GiB changed 1M decode by −0.09 t/s [1 run each]. Measured at 1M only.

---

## 10. Reference

### Branch state
```
branch  ssd-decode-bank-shrink
HEAD    b77b624
base    a04f46f  (origin/main now 6e4c285, no conflicts)
tree    clean, zero compiler warnings
```

### Code landmarks

`ds4_metal.m`
```
13291  ds4_gpu_stream_expert_nocopy_enabled
15157  "Residency for deferred expert dispatch"  (MTLResidencySet for the bank)
15530  "Deferred expert residency"               (state, backoff, token mode)
15584  ds4_gpu_stream_expert_miss_flag           (per-token miss word)
15646  DS4_METAL_DEFER_BACKOFF_MIN / MAX / FORGIVE
15732  ds4_gpu_stream_expert_defer_replay_routing
17604  batched pending loader (no-copy branch)
17924  load_selected_missing   (no-copy branch, e4b81b1)
18556  prepare_selected_batch  (no-copy branch, e4b81b1)
27131  ds4_gpu_dsv41_attention_output_batch      (hardcoded Q8_0 — Lead 1)
27154  ds4_gpu_attention_output_q4_K_batch_tensor (n_tokens < 32 early-return — Lead 1)
~41300 ds4_gpu_routed_moe_one_tensor (V4.1) — deferral decision lives here
```

`ds4.c`
```
 5539/5541/5641  tensor_expect_layout attn_output_* Q8_0   (Lead 1)
~40922  ds41_verify_snapshot_encode
~41044  decode-side Engram reads
~41068  ds41_engram_decode_bg   (background layer-14 read)
~41095  queue_layers / overlap / drain / blocking
~41210  ds41_graph_step         (deferred decode + rollback wrapper)
~64470  SSD auto cache plan
~67600  static weight mlock;  ~67930  support model mlock (276a3fe)
~78785  ds4_session_ds41_dspark_cycle
```

`metal/moe.metal`
```
~3974  kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32   (raises the miss atomic)
~6103  kernel_mul_mv_addr_q2_K_sum6_f32             (raises the miss atomic)
```

### Environment variables added on this branch

| var | effect |
|---|---|
| `DS4_METAL_STREAM_EXPERT_NOCOPY=1` | no-copy expert cache |
| `DS4_METAL_V41_DISABLE_DEFER_EXPERT_SYNC=1` | disable deferred expert residency (default: on) |
| `DS4_METAL_DISABLE_V41_DECODE_OVERLAP=1` | one command buffer per token instead of async per-layer |
| `DS4_METAL_MAX_ASYNC_BATCHES=N` | in-flight command buffer cap (default 8) |
| `DS4_ENGRAM_DECODE_SERIAL=1` | both Engram reads at the head of the token (old behaviour) |
| `DS4_ENGRAM_DECODE_PROFILE=1` | report Engram decode read cost |
| `DS4_DISABLE_SUPPORT_MODEL_MLOCK=1` | do not wire DSpark weights |
| `DS4_METAL_V41_DEFER_DEBUG=1` | print which gate blocked deferral |
| `DS4_METAL_MODEL_VIEW_DEBUG=1` | dump model-view table on coverage failure |
| `DS4_METAL_ROUTE_REPEAT=1` | expert routing repeat between tokens (measured 37% expert, 0.6% full-layer) |
| `DS4_V41_DSPARK_LOG=1` | per-cycle DSpark accounting |
| `DS4_METAL_V41_DEFER_REPLAY_PRUNE=1` | re-enable pruning in the deferred routing replay (measured worse) |

### Reproduction

Short-context A/B — the best-conditioned measurement on this machine:
```bash
P="Write one paragraph explaining how a B-tree differs from a hash index."
export DS4_METAL_STREAM_EXPERT_NOCOPY=1
for i in 1 2 3; do
  ./ds4 --ssd-streaming -m gguf/DeepSeek-V4.1-Flash-Q2.gguf -p "$P" -n 60 --temp 0 \
    2>&1 | grep -oE "generation: [0-9.]+" | sed "s/^/defer $i /"
  DS4_METAL_V41_DISABLE_DEFER_EXPERT_SYNC=1 ./ds4 --ssd-streaming \
    -m gguf/DeepSeek-V4.1-Flash-Q2.gguf -p "$P" -n 60 --temp 0 \
    2>&1 | grep -oE "generation: [0-9.]+" | sed "s/^/sync  $i /"
done
```
Observed: `defer ≈ 21`, `sync ≈ 18.4`, discarding run 1.

Correctness gate (must stay byte-identical):
```bash
P="Explain in detail how a write-ahead log guarantees durability, and why fsync placement matters."
./ds4 --ssd-streaming -m gguf/DeepSeek-V4.1-Flash-Q2.gguf -p "$P" -n 150 --temp 0 \
  2>&1 | grep -v "^ds4:" > /tmp/a.txt
DS4_METAL_V41_DISABLE_DEFER_EXPERT_SYNC=1 ./ds4 --ssd-streaming \
  -m gguf/DeepSeek-V4.1-Flash-Q2.gguf -p "$P" -n 150 --temp 0 \
  2>&1 | grep -v "^ds4:" > /tmp/b.txt
diff /tmp/a.txt /tmp/b.txt && echo IDENTICAL   # 3009 bytes at b77b624
```
This was run on two prompts across the branch (2913 B and 3009 B outputs). It is not a general equivalence proof — it is two prompts at 150 tokens.

Re-derive §3: read the GGUF tensor table, compute bytes/weight from **consecutive tensor data offsets**, scale `*_exps` by 6/384, sum. Do not assume type codes.

---

## 11. What I would not take on faith

1. **The 546 GB/s peak bandwidth figure is unverified on this hardware.** Every headroom conclusion depends on it. Measure it first.
2. **No genuinely full 1M-token context has been measured.** All "1M" figures are an allocated window with a short prompt.
3. **I derived the weight-traffic table wrong twice before §3.** The first error produced a "9% of bandwidth" figure I asserted to Neil and later retracted. Re-derive from file layout.
4. **Several results rest on single runs** — the support-model mlock throughput delta, the `nsg` sweep, the 65k arms, the cache-size test. They are labelled inline. The deferral result (§4.3) and the concurrency result (§5) are the best-evidenced.
5. **The engram overlap block's absolute throughput (12.6–12.9) is not comparable** to the §0 table; those runs were almost certainly contended by `ds4-server`. The delta is credible, the absolutes are not.
