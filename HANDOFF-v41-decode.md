# DeepSeek V4.1 Flash on ds4 / M5 Max 128 GB — Engineering Handoff

**For:** Astra (implementation)
**From:** Neil Gainey's ds4 session, September 2026
**Branch:** `gaineyllc/ds4` → `ssd-decode-bank-shrink` (latest `1f99c22`); upstream PRs #1059, #1060, #1061 open against `antirez/ds4`
**Goal:** DeepSeek V4.1 Flash, Q2 experts, 256k context, on one MacBook Pro M5 Max 128 GB, decode at 50 tok/s. Current: 14–17 tok/s with DSpark at 16k, 10–12 without.

This document is everything learned, measured, built, and rejected, in one place, followed by the design Neil is directing (ragged/sawtooth batching, JIT experts, draft-keyed prefetch) and an honest assessment of the pasted "Zero-Copy Engram Table Offload" spec against what ds4 already is. Read sections 2, 5 and 8 first if short on time.

---

## 1. Ground truth about the model and the machine

These numbers were checked against the GGUF headers, the HF release, and ds4's loader. Several documents floating around (including the pasted spec) have them wrong.

| Item | Value | Note |
|---|---|---|
| Layers | 40 | all routed after the dense stem |
| Routed experts per layer | **384** (top-6 used) | not 256 |
| Routed expert size at Q2 (gate+up+down) | 9.49 MiB | IQ2_XXS gate/up, Q2_K down |
| Routed experts total at Q2 | **142 GiB** | this is the thing that does not fit |
| Engram n-gram tables | 189 GiB, FP8, two tables of ~94 GiB | at the end of the GGUF; **already disk-only in ds4** (pread, 24 scattered rows per table per token) |
| DeepSeek release format | FP8 dense + **FP4 experts** (`expert_dtype: fp4`) | there are no BF16 experts to "go get" |
| antirez's `ds41f-q4` | Q4_K gate/up + MXFP4 down | MXFP4 = bit-exact repack of the released FP4 |
| Expert bank at 92 GB budget | 8509 entries = **55 % residency** | the number that governs everything below |
| DSpark drafter | 3-stage MTP, block 5, `DeepSeek-V4.1-Flash-DSpark-native.gguf` | fully resident, never streams |
| Machine | M5 Max, 128 GB unified, NVMe | ~5–6 GB/s sustained expert reads observed; SSD 80–95 % idle during decode |

The single sentence to keep in mind: **the GPU can only run a layer once all six experts every row routes to are in memory, and only 55 % of them are.** Every result in this document is downstream of that.

---

## 2. What ds4 already does (state of the branch)

ds4 is not a naive streamer. Before designing anything, know that these exist and work:

**No-copy expert bank.** Experts are `bytesNoCopy` Metal buffers over the model's own mmap pages, held in one `MTLResidencySet`. An expert enters the bank by creating a view and adding it to the set (no memcpy); the residency commit wires the file pages. Copy mode (pread into GPU slabs) still exists and is faster per token at small budgets (+20–30 % decode at 30 GB) but halves the bank you can afford. `DS4_METAL_STREAM_EXPERT_NOCOPY=1` selects no-copy.

**Address-table MoE kernels.** The routed kernels (`kernel_mul_mv_*` for decode rows, `kernel_mul_mm_id_*_cached_*` with `EXPERT_ADDRESSES=true` for ≥64-row batches) reach experts through a per-layer GPU address table, so the dispatch does not need to name six buffers per row and the host does not need to read the router back before encoding.

**Deferred expert sync (the "loop-around" already exists).** A verify batch runs all 40 layers without the host reading the router. The cached-address kernels check residency on the GPU and set a miss flag. If anything missed, the host installs the missing experts and **redoes the whole batch from layer 0**, then backs off into stop-per-layer mode for a cooldown (`DS4_METAL_DEFER_BACKOFF_MIN/MAX`, capped by `DS4_METAL_V41_DEFER_BACKOFF_MAX`). Entry points: `ds4_gpu_stream_expert_defer_begin_token`, `ds4_gpu_stream_expert_defer_token_missed`, `ds4_gpu_stream_expert_defer_replay_routing`.

**DSpark speculative decode.** Drafter proposes ~3 tokens; the verify batch runs the drafted rows plus the anchor through the main model in one pass; longest agreeing prefix commits; rollback ring restores KV/position/Engram state. Output is bit-identical to greedy. Measured: acceptance 71–74 %, **2.01–2.19 committed tokens per cycle**, verify batch 150–193 ms at 16k.

**Bank prefill (built this session).** Agent-turn prefill (short tail after a warm prefix) takes experts from the bank via a grouped matmul over the address table instead of mapping every layer. Agent-turn prefill went 46→17 s (1080 rows), 74→32 s (3770 rows), 80→40 s (5534 rows), output byte-identical. Mechanics: `bank_transient` eviction priority so prefill does not churn the decode bank, whole-tail sweep to 8192 rows, async `F_RDADVISE` from a helper thread (`ds4_gpu_stream_expert_readahead_ranges_async`).

**Hotness / eviction.** Frequency-with-decay counters per (layer, expert) decide which 55 % stays resident. It is not a predictor; it is the LRU/LFU policy. With it suspended during bank prefill the bank churned to 72–76 % miss and decode fell to 2.4 t/s; with it on, ~50 % hit is what you get. No clean hotness-vs-random A/B exists.

**Disk KV store, live-prefix rewind, ds4-bench replay.** Prefix replay past ~229k (snapshot >1 GiB) recomputes the prompt; it is prefill, not prediction.

**Fixes landed this session.** Residency-set leak (evicted no-copy views handed to `reuse` while still set members: 15315 views / 47.5 GiB for 2468 entries → 7488 views, wired flat at 47–49 GB; this was the 65k OOM). Radix top-k determinism (PR #1060: select per visible width 4096, split straddling batches, tie-count pass, ties by index). Loader support for mixed GGUFs whose tensors trail the Engram tables (`hole_start/hole_end`).

**Tools.** `gguf-tools/deepseek41_native_expert_layers.py`: APFS-clone a Q2 GGUF and replace chosen layers' experts with the released FP4 repacked bit-exact as MXFP4 (appended payloads, patched records). Built `DeepSeek-V4.1-Flash-Q2-L37-39native.gguf`.

---

## 3. Everything measured (so nothing gets re-measured by accident)

| Experiment | Result | Verdict |
|---|---|---|
| Bank prefill, agent turns | 46→17 s / 74→32 s / 80→40 s, identical output | **keep** |
| Split-miss (run resident rows, redo missed) | 0 gain | stashed, not committed |
| Edge0 prerouter, zero-training proxy (layer L+1 router weights on layer L state) | 22.6 % recall of misses | not worth prefetching on; needs a trained head |
| DSpark acceptance vs quant (attn Q4_K, dense Q4_K, head, native experts L37–39) | flat 71–74 %, 2.0–2.2 committed/cycle | **drafter is not misaligned to the quant**; experts are not the problem either |
| Speed track, no-copy 30 GB vs upstream main | ≈ main prefill, −5 % decode | |
| Speed track, copy mode 30 GB | +20–30 % decode, −5–10 % prefill | copy mode should be the small-budget default (bisect of prefill cost still open) |
| Speed track, no-copy 92 GB | −15–20 % prefill, decode ≈ main (no DSpark in ds4-bench) | |
| Previous-token predicted preload (`DS4_METAL_V41_PREDICTED_LOAD`, pre-existing) | −7 % at 8k | opt-in only |
| madvise WILLNEED warmer (pre-existing) | 6.36 vs 10.00 t/s in copy mode | off by default; wrong mechanism when the bank holds copies |
| **Draft-token expert hints** (this session, see §6.4) | 15.36 → 14.58 t/s (−5 %); prefill 186 → 169 t/s | recall not yet measured; default should flip to off until it is |
| **`DEFER_NO_BACKOFF`** | 15.36 → 10.65 t/s (−30 %) | dead; leave as opt-in knob |
| 256k regression tracks | correctness tracks pass with `DS4_TEST_SSD_STREAMING=1`; `logprob-vectors short_code_completion` step-0 mismatch is identical on upstream main (quant vs API) | |

Config for the 16k numbers: 10k prompt, 400 generated tokens, `-c 16384`, `--ssd-streaming-cache-experts 36GB`, no-copy, DSpark, `~/ds4-bench/ab16k.sh`.

---

## 4. Where the time goes (the decode limiter)

At 16k with DSpark a cycle is ~100 ms wall: verify 150–193 ms per batch over ~2.1 committed tokens, plus draft ~7 ms and gap ~8 ms. The floor for one verify batch with everything resident is ~45 ms (streaming the resident weights out of unified memory for 4–6 rows; row count barely matters because expert bytes dominate). The other ~55 ms per cycle is **expert-miss stops**: GPU drain, host reads the miss set, SSD read (~2 ms per expert at 9.5 MiB), residency commit, clock ramp, re-encode, and on the deferred path a full redo from layer 0 followed by a stop-per-layer cooldown.

Two things follow. First, no batching trick changes the 45 ms; only memory bandwidth does. Second, everything that attacks the 55 ms is one of exactly three levers: **fit more of the bank** (memory), **know the routing earlier** (prediction), or **make each stop cheaper** (engineering). No-copy only touches the cost of getting a missing expert in; it does not change the batch or the stall.

Why prefetching is hard here specifically: layer L's routing is a matmul on a hidden state that layer L−1 has not produced yet, and within a batch row t's attention at layer L needs row t−1's layer-L KV, so a miss at (row r, layer L) blocks every row ≥ r. Nothing recorded from the past gives you next-token routing directly. The drafter changes this: it gives you the **token ids** of the next verify batch before it runs, which is a real prefetch window (§6.4).

---

## 5. Neil's proposals, assessed

### 5.1 Staggered / overlapping verify cycles that "loop around" on a miss

The loop-around exists (deferred path, §2). What does not exist and cannot exist without speculation is overlapping two verify cycles: cycle n+1's draft is a function of cycle n's committed token and hidden state. Until n commits there is nothing to run. Tree speculation (Medusa / EAGLE-2 / SpecInfer: several draft branches per position, verified together under a tree mask, longest agreeing branch wins) is the legitimate form of "run more candidates concurrently", but a 16-node tree touches up to 96 experts per layer instead of 24 and at 55 % residency that means more misses per cycle. It pays when the bank fits; it hurts here until residency is fixed.

What is still on the table inside the existing loop-around, unbuilt:

1. **Redo from the missed layer, not layer 0.** Checkpoint the residual after each MoE layer during the deferred pass; on miss, install and resume at the lowest missed layer. Roughly halves redo cost. Needs a per-layer residual checkpoint in `ds41_graph_prefill_sweep` for verify batches and a resume entry point.
2. **Issue the miss reads before the drain finishes.** The miss buffer is host-visible; a helper thread can poll it and issue `F_RDADVISE` the moment a layer writes it, the way bank-prefill readahead already runs.
3. Do **not** remove the backoff (measured −30 %).

Expected total from 1+2: single digits to low teens percent. Not 2×.

### 5.2 "Run the rows as concurrent threads, most accurate wins"

Rows already run concurrently on the one GPU in one command buffer. Correctness is not a vote: the main model's output at row t−1 *is* the grade for row t. The concurrent-candidates version of this is tree speculation above.

### 5.3 JIT experts / evict the coolest to make room

That is what a miss install does now (victim = lowest hotness, `bank_transient` first). The bank is always full; every install is an eviction. What "JIT" adds only if the routing is known early enough to install *before* the layer needs it, which returns to prediction.

### 5.4 KV + replay + experts as a lookup graph → draft-token hints (built, measured once)

The useful version of the graph idea: per-(layer, token id) table of the experts that token routed to before, fed by the same routing readback that feeds hotness; when a verify batch is announced, read ahead the predicted experts for every row and layer that are not in the bank. Built as `ds4_gpu_stream_expert_hint_rows()` and friends in `ds4_metal.m` (search "Draft-token expert hints"), hooked in `ds41_verify_suffix_tops_once` and `ds41_graph_step_once`. Prefetch-hint only (no installs, no evictions, cannot change output). Env: `DS4_METAL_V41_HINT=0|1|2`, `_TOPN` (3), `_MIN` (2), `_MAX` (128). Memory ~124 MB for 40 layers × 129k vocab.

Measured once: −5 % decode, −9 % prefill. The recall line (the number that decides whether the table is any good) did not print on the CLI path; an at-exit report was added (`ds4_gpu_stream_expert_hint_report_atexit`) but the rerun hung under a bad machine state (§7). **Do this first after a reboot:** run `ab16k.sh` on/off, read the recall. If recall < 30 %, try `TOPN=6 MIN=1` once; if still low, flip the default to off and keep the table only as training data for a real prerouter. If recall is high and it is still a loss, the readahead is competing with the stops for the SSD and needs to be throttled to layers ≥ N ahead of the sweep.

For prefill/replayed tokens the routing *is* deterministic: storing expert ids per token per layer beside the disk KV gives exact prefetch on replay. Smaller win because prefill is already layer-major with batched misses; not built.

### 5.5 The ragged / "sawtooth" batcher (Neil's original direction)

Within one verify batch, rows are already grouped by expert: the address-table kernels run one dispatch per layer over all rows, and the ≥64-row cached `mm_id` path (bank prefill) is a true grouped GEMM per expert. So single-session decode already has the batching the pasted spec asks for, at 4–6 rows, which is far too few rows for GEMM to beat matvec. That is the sawtooth: prefill runs the fat GEMM path, decode runs the thin matvec path, and the cost per token is set by expert bytes either way.

Where ragged batching **does** buy something is across sessions. If N sessions decode concurrently, their rows at layer L can be binned by expert and each expert's weights read once for all of them. Two things to be clear about before building it:

- It raises **throughput**, not per-session latency. Each session still waits for the full 40-layer pass; the pass gets a little longer, not shorter.
- It **raises the expert working set per pass**. Six sessions' verify rows can touch up to 6×24 = 144 distinct experts per layer instead of 24; at 55 % residency that is more misses per pass, and misses are what already cost half the cycle. Cross-session batching helps most when the bank fits and hurts when it does not, exactly like tree speculation.

So on this machine the ordering is: fix residency or stop cost first; then ragged batching is a throughput multiplier for the server, which is a different goal from 50 tok/s on one stream. Both are worth having; only one moves the number Neil is chasing.

---

## 6. The pasted spec ("Zero-Copy Engram Table Offload & MoE Execution Engine") against ds4 reality

Take the spec as a good description of the shape of the problem and a poor description of where the bytes actually are. Point by point:

**Engram is not the memory problem.** The spec treats the 183 GiB Engram table as the thing that has to be offloaded. In ds4 it already is: the tables are unmapped and read with `pread` at the head of every token, 24 scattered rows per table, ~32–64 KB per token, off the critical path. The thing that does not fit is the **142 GiB of routed experts at Q2**, of which a 92 GB bank holds 55 %. The spec's MoE line ("~60–75 GB, pinned resident") is what a Q2/Q4 *dense-plus-routed* footprint looks like only if you ignore that 384×40 experts cannot all be resident. Rewrite section 1 of that spec around experts, not Engram.

**Zero-copy mmap + `newBufferWithBytesNoCopy` for Engram.** Feasible, and it is exactly the mechanism ds4 already uses for *experts*. For Engram it is optional: 48 rows per token via pread is not a stall we have ever measured. If Astra wants the fused gather-gate kernel for cleanliness, fine, but do not expect a decode win from it. Note the constraint the spec omits: a file-backed no-copy buffer is only safe to read from the GPU once its pages are resident (residency commit wires them); "madvise WILLNEED then dispatch" is not a barrier. ds4 handles this with the residency set plus pending/retired member tracking; the leak fixed this session lives exactly there.

**Two-phase CPU prefetch with `MADV_WILLNEED`.** For Engram: the hashes are deterministic at tokenization, so yes, this is trivial and free. For experts: the prior `MADV_WILLNEED` warmer in ds4 measured 6.36 vs 10.00 t/s in copy mode (double-populates memory). In no-copy mode `F_RDADVISE` from a helper thread is the right primitive and is what bank prefill and the draft-token hints use. The spec's claim "eliminating pipeline stalls" is only true when the addresses are known ahead of time, which for experts they are not (§4).

**Ragged batching / grouped GEMM.** Already present for ≥64 rows (`kernel_mul_mm_id_*_cached_*`, `DS4_METAL_V41_BANK_MM_MIN`). For decode rows it is matvec by design. The cross-session form is §5.5, with the residency caveat.

**"Full tensors, never partially sliced."** Correct and already the case: an expert is gate+up+down for one (layer, expert), 9.49 MiB, installed or evicted as a unit.

**Roadmap Phase 1 (re-serialize Engram into a 64-byte-aligned FP8 file).** Unnecessary; the GGUF tables are already contiguous and page-aligned enough for the pread path, and the loader now tolerates tensors trailing them. Skip unless the fused kernel is built.

**Roadmap Phase 3 (`MTLSharedEvent` ring between CPU worker and GPU queue).** This is the right structure for the "issue miss reads before the drain finishes" item in §5.1, and for a multi-session server. It is not needed for Engram.

Net: about a third of the spec is already built in ds4 under different names, a third is aimed at the wrong tensor, and a third (cross-session ragged batching, CPU/GPU event ring, per-layer resume) is the actual work.

---

## 7. Operational facts that cost a night

- **Build only from a macOS shell.** Running `make` in the Linux VM that mounts the repo pollutes the tree with ELF `.o` files.
- **`DS4_METAL_STREAMING_BATCH_PROFILE=1` is for the server.** It was believed to kill the CLI; it was not the cause, but do not use it on `ds4` CLI runs. Stats lines for hints print at exit; `DS4_METAL_MEMORY_REPORT=1` is not on the V4.1 CLI path.
- **`~/ds4-bench/watch2.sh` kills ds4 on swap > 8000 MB** (raised to 24000 for the last runs). After several SIGKILLed streaming runs the machine sat at **13 GB wired, 2 GB compressed, 9.7 GB swap with nothing running**: leaked wired no-copy pages / residency sets. That state made runs die silently at startup and finally hung one in `ds4_gpu_wait_command_buffer` (state UN, 0 % CPU). **Reboot before measuring anything.** Prefer letting runs finish over `pkill -9`.
- **Auto budget picks 55 GiB (6003 entries) when memory looks free**; pin `--ssd-streaming-cache-experts 36GB` for A/B runs so the bank is constant.
- Bridge sleeps ≥ ~55 s time out; poll ≤ 45 s.
- Do not delete Neil's Q2 side-variant GGUFs without his say-so (four × 366 GB nominal; disk ~79 GiB free).

Harness: `~/ds4-bench/ab16k.sh <tag> [ENV=VAL…]` (16k single-stream), `bp_run.sh <tag> [ENV…]` (server, 256k, four agent-turn requests), `bench256.sh <tag> <dir>` (ds4-bench sweep, `BUDGET/CTXSTART/CTXMAX/NOCOPY`), `acc_*.txt` acceptance logs, `probe1.txt` Edge0 probe, `bench/PLAN-50tps.md` dated notes for every step.

---

## 8. Recommended plan for Astra, in order

1. **Reboot. Rerun hints on/off, read recall.** Decide the hint default from the number (§5.4). Half a day.
2. **Redo-from-missed-layer + early miss reads** (§5.1 items 1–2). Measure at 16k and 64k. Expect +5–15 % decode. This is the only stop-cost work not yet tried.
3. **Copy mode as the small-budget default** and bisect its −5–10 % prefill cost (candidates: prefill pins, the 7.12 GiB prefill expert reserve, end-of-sweep seed). Cheap win for anyone on a 64 GB machine.
4. **Trained prerouter head** (Edge0's real mechanism): per-layer small head predicting layer N+1 routing at token t+1, trained offline from ds4 routing logs (the hint table's readback is the data source; add a dump). Use it only as a prefetch hint, never as the routing (Edge0's "prediction is routing" needs a recovery LoRA and costs quality). Recall must clear ~50 % of misses to matter. This is the one lever that can turn stops into overlap without more memory.
5. **Cross-session ragged batching** for the server (§5.5) once 1–4 are in, with a hard cap on distinct experts per layer per pass so it cannot drive the miss rate up.
6. **Do not** retry: split-miss, previous-token predicted preload, WILLNEED warmer in copy mode, no-backoff, quant-alignment of the drafter, native experts as a speed fix (it is a quality knob only).

The honest ceiling with 1–4 on this machine is roughly 20–25 tok/s single-stream at 16k. 50 tok/s single-stream needs the routed experts to fit (a 192–256 GB machine, or a 2-bit-and-below format that halves 142 GiB without wrecking acceptance), or a prerouter good enough that stops disappear. Neither is engineering alone.

---

## 9. Index of code touched this session

`ds4.c`: `ds41_gpu_graph.bank_prefill`; session-sync chunk loop (`bank_max/bank_total`, `bank_eligible`); prerouter probe (`DS4_V41_PREROUTE_PROBE=1`, `ds41_probe_*`); loader Engram hole (`hole_start/hole_end`); hint hooks in `ds41_verify_suffix_tops_once` and `ds41_graph_step_once`.

`ds4_metal.m`: bank prefill begin/end and admission; `use_bank_mm` path in `ds4_gpu_routed_moe_batch_tensor`; `bank_transient` + victim hotness; `ds4_gpu_stream_expert_readahead_ranges_async`; `clear_entry_internal` reuse-path fix; owned-view counters and `ds4_gpu_stream_expert_bank_report`; draft-token hint module (table, `hint_note`, `hint_rows`, at-exit report); `DS4_METAL_V41_DEFER_NO_BACKOFF`.

`metal/moe.metal`: `kernel_mul_mm_id_q2_K_cached_f16(_mpp)`. `metal/argsort.metal`: tie-count/compact top-k (also on `pr/indexer-topk-select`, `be6a8ce`).

`gguf-tools/deepseek41_native_expert_layers.py`: mixed-file builder.

Commits, newest first: `1f99c22` hints results + at-exit report; `84f72cd` draft-token hints + no-backoff knob; `3a6f716` mixed native-expert files + loader; `2ab8d79` acceptance flat; `be5b6b6` final sweep table; `5d6cb16` residency leak fix; `e76188a` probe; `dbdb91e` top-k fix; `2dffde7` readahead thread; `bd972fb` whole tail; `1b6df89` bank prefill. Backup branch `backup/ssd-decode-bank-shrink-pre-rebase-*`.
