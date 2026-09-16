# ds4 / V4.1 Flash Q2 — plan to 50 t/s (2026-09-14, Fable 5.1)

## Measured this session (branch ssd-decode-bank-shrink @ 550c944, rebased on origin/main 9139e2a)
- Achievable Metal read bandwidth, M5 Max (bench/membw.m, 16 GiB, 7 warm runs each):
  coalesced (consecutive threads -> consecutive 16 B): 570 GB/s mean, 575 best
  per-thread contiguous chunk (matvec-row style):       390 GB/s mean, 394 best
- Weight bytes per decoded token, from GGUF offsets (bench/gguf_traffic.py): 10.50 GB
  (handoff said 9.79; it omitted output.weight 0.70, ffn_gate_inp 0.32 F32, attn_q_a 0.28, attn_kv 0.11, indexer 0.08).
  attn_q_b + attn_output_a + attn_output_b = 4.99 GB = 47.5%. All Q8_0.
- GPU busy per token (DS4_METAL_GPU_BUSY_PROFILE, 148 tokens, ctx 32k, short prompt, machine in interactive use):
  32.5 ms busy per token; 80 command buffers per token; wall ~61 ms (16.4 t/s contended; 20-22 uncontended per handoff).
  => GPU idle ~40-45% of every token. While busy: 10.5 GB / 32.5 ms = 323 GB/s.
- Ceilings: 10.5 GB/token at 570 GB/s = 54 t/s; at 390 GB/s = 37 t/s. 50 t/s = 20 ms/token.
  Conclusion: 50 t/s is NOT reachable by bandwidth efficiency alone at current byte volume. Need bytes down AND idle out.

## Workstreams, in order
### W1. Attribute the idle 40% (measure first, 1-2 days)
  DS4_METAL_ENCODER_TIMELINE + host-side timestamps around: engram reads (2.2 ms/token measured), expert address install,
  ds41_bf16 rounding dispatches (9.7/layer), per-layer command-buffer commit (80/token), sampling/readback at layer 39.
  Deliverable: a per-token waterfall. Hypothesis to test: 1 command buffer per token with GPU-resident routing and the
  sampler on-GPU removes most of the gap; the handoff's "1 cb/token slower" result predates full deferral and was contended.
### W2. Q4_K attention projections (bytes -4.99 -> -2.64 GB; 2-3 days)
  No 340 GiB rebuild needed: `cp -c` (APFS clonefile) the GGUF, patch the three tensor type ids in the header,
  dequantize Q8_0 -> f32 -> ds4q_quantize_chunk(Q4_K) (gguf-tools/libds4quants.so), write back at the same offsets
  (Q4_K rows are smaller; gaps are fine, ds4 sizes tensors from type+dims). ~3 GB COW delta. All in_dims are multiples of 256.
  Code: relax tensor_expect_layout at ds4.c 5993/5995/6091/6095/6096; ds41_attention_low -> ds4_gpu_attention_output_low_q4_K_slice_tensor
  (exists, ds4_metal.m 28087, used by V4 Flash); ds41_matmul already dispatches dense quant via metal_graph_matmul_plain_tensor;
  prefill batch path at ds4.c ~42917 already has the q4_K batch tensor kernel. Validate on real review prompts, not perplexity.
  Then extend to shared experts (1.5 -> 0.8 GB) and output.weight (0.70 -> 0.37). Floor with everything non-routed at Q4_K: ~6.8 GB/token.
### W3. Coalesced matvec (390 -> 570 GB/s class; 2-4 days)
  kernel_mul_mv_q8_0_f32 is 35% of kernel time. The packed_char4 attempt changed instruction count, not access pattern.
  Restructure so a simdgroup reads one row cooperatively (lane i reads 16 B at i*16) — the access the 570 figure measures.
  Same for the Q4_K decode kernels W2 lands on. Byte-identical arithmetic is possible (same reduction order per row) — keep the gate.
### W4. Multi-row decode at near-single-row cost (unlocks DSpark AND concurrency; 3-5 days)
  Measure decode cost for 1/2/4/8 rows through the same layers. Bandwidth model says 5 rows should cost ~1.6x one row
  (attention weights read once; up to 30 unique experts/layer). Measured DSpark verify was 52 ms/row = 3.4x worse than that model
  => the verify batch path is not bandwidth-shaped. Fix that path and DSpark's ceiling moves from 1.5x to ~3x at current acceptance,
  and ds4-server can batch concurrent slots instead of time-slicing (Lead 4: aggregate flat at 11 t/s).
### W5. Long context (required for the 1M goal; measure before designing)
  65k occupied: 9.5-10.2 t/s, every token misses the expert bank. Nothing above 65k has ever been measured.
  Run 128k / 256k / 512k occupancy with the profile env vars; separate KV/attention cost from expert-cache thrash.
  Also: entries=6611 > budget=5973 and live=61.28 > target=55.37 GiB in this session's stats — check the accounting.

## Arithmetic target
  bytes 6.8 GB (W2) at 500 GB/s (W3) = 13.6 ms busy; idle cut to <5 ms (W1) => ~18-19 ms/token => 52-55 t/s single-stream at short context.
  W4 multiplies whatever that lands at for served throughput.

## Hygiene
- Server respawn: ~/bin/hermes-restore-customizations.py runs `launchctl load -w` on com.ds4.server (re-enables it);
  ~/bin/ds4-auto-build.sh bootstraps+kickstarts it. Both re-arm the server after bootout/disable. Decide what to do with them.
- Stale `vi .git/COMMIT_EDITMSG` (pid 4989) from a 2026-09-13 `git rebase --continue` shell is still open; harmless so far.
- Fixed: expert-cache residency-set leak (550c944). tests/test_metal_ssd_experts passes; make test's other failures are
  environmental (ds4flash.gguf -> V4.1 Q2 but the golden vectors are V4 Flash; ds4_engine_open without --ssd-streaming;
  ds4_server.c:17303 think-mode assert is upstream code, untouched by this branch).

## Session log (continued, 2026-09-14 afternoon) — measured
- Early expert load drained the GPU on EVERY layer even under deferral: 36.8 ms/token host wait. Now skipped for
  deferred tokens; deferral starts only after 4 clean stopping tokens (warm evidence). Residency commits: 17/token -> 1/token.
- SSD cold read: 8-12 GB/s (1.2 ms per 9.5 MiB expert). A 200-token answer touches ~7600 distinct experts (72 GiB);
  bank max ~7600 (16k ctx) / ~6300 (1M ctx) experts. => long outputs are SSD-bound (6-12 t/s) at 384 experts/layer. Physics.
- Fix chosen (agency granted): DS4_EXPERT_KEEP per-layer keep-list -> router bias mask; experts outside the list are never
  routed/read. Bank holds the kept set, deferral never misses. Keep-list K is set by memory: K <= bank_slots/40
  (~190 at 16k ctx, ~157-175 at 1M ctx). DS4_EXPERT_USAGE_DUMP collects the (layer, expert) histogram on the GPU.
  Coverage of routing weight by top-K (4 code-review prompts, prefill+decode): K160 91%, K192 94%, K224 97%, K256 98%.
- Resident regime, K=128 keep, warm server, 200-token requests: 27.6 t/s, GPU busy 88% (31.8 ms/token). GPU-bound now.
- Q4_K in-place requant tool (gguf-tools/deepseek41_requant_dense.py, APFS clonefile): attnQ4K (q_b/out_a/out_b) and
  denseQ4K (+ q_a, shexp) variants built. Engine accepts Q4_K attn_output_a/b for V4.1 (decode via existing q4_K low kernel).
- Per-kernel (timeline, relative): routed expert kernels (iq2_xxs pair swiglu, q2_K sum6) run at ~150 GB/s effective
  vs ~350 for attn_out_low_q8 -- the IQ2_XXS/Q2_K expert matvecs are the least efficient bytes. Candidate: Q2_K gate/up
  (cheaper dequant, +27% expert bytes) or kernel work.

## Findings 17:30 (after correction)
- INVALID: 43-48 t/s runs with N_R0 shader-only changes (host dispatch computed threadgroups with the old row count;
  half the rows uncomputed). Fixed by injecting DS4_METAL_N_R0_* from the host into the Metal preamble. With correct
  outputs, 1/2/4 rows per simdgroup are within noise. Byte-identical gate is mandatory for every kernel change.
- Honest resident number: 35.7 t/s (denseQ4K, K=176 keep-list seeded, engram pool). GPU ~25 ms/token vs ~13 ms byte floor;
  remainder is latency-bound small kernels (hc F16 matvecs 106/token, router argsort 48/token, bf16 rounding 780/token
  measured at only 0.86 ms total, flash-attn vec at short ctx ~2 ms).
- QUALITY: keep-list K=176 (54% pruned) breaks the model: 9-10 of 12 core eval answers degenerate into repetition loops,
  with Q2 or denseQ4K alike. Pruning by usage at this ratio is not viable. K=224 (max at 16k ctx with 92% wiring) and
  a K=384/376 masking sanity check are queued; full-model baseline running.
- Conclusion for 128 GiB single node, full quality: long outputs are SSD-miss bound (bank ~7900 slots ~ 94% hit,
  ~13 misses x 1.2 ms per token) => ~20-25 t/s at 16k ctx, ~12-15 t/s at 1M ctx (bank ~6300 slots).
  50 t/s at full quality on this model needs the experts resident: second M5 Max over TB5 (TP splits experts and heads,
  halving bytes/token per node; deferral must be made TP-safe: g->tp_world != 2 gates) or a model that fits.

## 18:50 — OOM root cause, and the shape of the answer
- "layer 0 failed at position 555" in the full-model eval = kIOGPUCommandBufferCallbackErrorOutOfMemory. Cause: the eval
  ran with DS4_SSD_CACHE_AUTO_PCT=100 (no decode-bank shrink) and the bank churned; the pre-session binary OOMs even
  earlier (prefill of the 2nd question) under the same setting. Not a leak from this session; the decode shrink is a
  necessary safety margin whenever the bank churns. Keep-list runs (no churn, seeded set 65 GiB) ran 12 questions fine.
- Two-node (TB5 TP) is the credible path to 50 t/s at full quality: each node holds its half of the experts fully
  resident (71 GiB), per-node bytes/token ~5.5 GB, plus ~4-8 ms/token of per-layer exchanges. What the engine needs for
  that: an "all-resident" decode mode = keep_seed for every owned expert at startup + address-table kernels with no early
  loads, no drains, no snapshot/redo (deferral is currently gated off under tp_world==2).
- Multi-row decode (ds4-server --batched-session N, DSpark verify) runs ~5x slower per token than single-row; it is the
  prerequisite for both concurrent serving and speculative decoding and is untouched.

## Sep 14/15 — MTP (DSpark) at 16k, no pruning (Neil's redirect)
Same 77-token prompt, 300-700 generated tokens, -c 16384, DS4_SSD_CACHE_AUTO_PCT=62 unless noted. Plain decode 15-18 t/s.
- DSpark 4.3 -> ~20 t/s (short prompt), 13 -> 16.7 t/s with an 11.6k-token prompt (plain 10.7 there). Commits:
  2fdef63 (drafter views, verify pipelining, residency-set coverage, carry restore for odd commits),
  971192a (bank bound under multi-row decode + readahead), 39db472 (reference anchor layout, window keys for every
  position, confidence-scheduled verify length), 4148653 (row-sharing kernels, draft early stop, compute-copy).
- Root causes found: (1) drafter HC kernels ran on the 8k-row prefill buffer (60 ms/draft); (2) verify drained the GPU
  twice per layer and useResource'd every expert per submission (driver 2-6 ms/layer); (3) the multi-row path never
  pruned the no-copy bank -> 9.7k entries / 90 GiB wired -> swap thrash mid-run (also the likely eval OOM);
  (4) odd-length commits discarded (31-44% of cycles); (5) draft block anchored one row off vs DeepSeek's model.py;
  (6) window keys never written for prompt/fallback positions; (7) indexed attention for 2-6 rows ran 8-16 workgroups
  (0.7 ms/layer at 10k ctx) -> use the 12-way split kernel like single-token decode does.
- Acceptance is Q2-bound, not a bug: drafter confidence head predicts 0.78 for row 0, target accepts 0.57; where the
  drafter is >=0.8 sure it is right 84% of the time. Same numbers at the first DSpark commit (43aa82e) on this prompt.
  Confidence-scheduled verify (min survival 0.6) keeps ~2.1 tokens/cycle at ~70 ms/2-row sweep.
- Where a 2-row verify goes (short ctx, per layer): GPU 1.05 ms (routed 0.32, attention+hc 0.72), sync 0.4, misses
  0.2 x 1.8 ms, prepare 0.1. Sweep 70 ms + draft 8 ms per ~2.1 tokens. At 11.6k ctx: +~1.3 ms/layer attention before
  the split fix, ~+0.5 after.
- Tried and rejected: spinning on cb.status (slower), userspace page-touch prefault of misses (20 ms/expert), attnQ4K
  target (no gain at this batch size), decode-bank pct 80 (same as 62: the 7606 dynamic cap binds).
- Bank: pct 50 -> 62 (6039 -> 7488 entries, 98 GiB wired incl. everything) took 700-token DSpark 17.2 -> 20.2 t/s
  with no swap growth on this machine; 50 stays the default in code.

## Sep 15 — no-copy bank memory (two panics), miss cost floor, 16k numbers
Prompt: README + engram source, 5723 tokens (the earlier "11.6k" figure was a miscount); also a 14.4k-token prompt.
DSpark, -c 16384, 400 tokens, DS4_SSD_CACHE_AUTO_PCT=50 unless noted. All variants byte-identical output.
- PANICS: two watchdog-timeout kernel panics (memory starvation). Under DS4_METAL_STREAM_EXPERT_NOCOPY the
  layer-major prefill still seeded its keep-lists through the GPU-copy loader into mlocked slabs (~70 GiB, "could not
  mlock all buffers... locked so far 69.70 GiB"); the decode cap then evicted those entries without freeing anything and
  every miss added a view on top: wired = 70 GiB dead slabs + bank + 24 GiB static, whatever the pct. pct 62 "worked"
  before only because prompt-era hotness never let the slab entries go; the hotness-decay fix (f35fec9) let them go
  and the growth started at once. Fix 686bd53: the seed makes views too (no slabs in no-copy mode), the decode cap is
  applied before prefill, seeded pages are pinned during the sweep (residency publish every 4 layers), seed once
  after the last sweep. Wired: 82 GiB at pct 50 (56 GiB bank), 91 GiB at pct 58 (65 GiB bank). Baseline OS wired is
  ~12 GiB; the machine has ~20 GB of other apps, so pct 58-62 leaves little slack -- 50 stays the code default.
- Hotness aging for the multi-row path (f35fec9): misses/layer 1.8 -> 0.8, verify 118 -> 98 ms, 15.8 -> 18.3 t/s.
- Numbers (5.7k prompt): overall 18.3 t/s (pct 50) / 19.6 t/s (pct 58); steady state 22-24 t/s. 14.4k prompt: 15.8
  overall, 18-22 steady (attention +0.1 ms/layer). First ~45 cycles are slower (1.5-2 misses/layer: the seed's
  keep-list is only ~78% of what decode routes to).
- Per layer, 3 rows, steady (CB_TIMES + batch prof): GPU 1.5 ms (cbA routed 0.5-0.6, cbB attention..router 0.9-1.0),
  host turnaround 0.21 ms with no miss, and +1.36 ms per miss-layer (view creation ~0.3, driver page-in ~0.85 of which
  SSD ~0.65). Sweep ~97 ms = 60 GPU + 8 turnaround + ~30 misses; draft 7.7 ms (5.8 GPU: 3 stages + vocab head for
  5 rows; 1.9 Markov loop: one GPU round trip per row). 2.4 tokens/cycle -> 22-23 t/s.
- Misses are the SSD: pread microbench 0.6-0.7 ms per expert (9.5 MiB, ~15 GB/s aggregate, 0.29 ms from page cache).
  Tried and rejected, all byte-identical: (a) parallel warm preads into scratch before the driver page-in: driver
  880 -> 242 us/miss but +1 ms host -> 13.8 t/s; (b) wired, resident miss pool with sync preads: prep 0.4 -> 1.05 ms
  (I/O now on the host) -> 111-122 ms verify; (c) same pool with GCD async preads and an MTLSharedEvent wait before the
  routed dispatch: host prep 0.25 ms but drain +0.5 ms per miss -> 108-124 ms verify. The driver's page-in is already
  the cheapest way to move a miss; the only miss lever left is fewer misses (bank size) -- patch kept at
  bench/miss-pool.patch on the machine for reference.
- Prediction is dead: DS4_METAL_ROUTE_TRACE dump over 281 cycles: misses are not predicted by the previous layer's
  experts (L-1 -> L transition table, top-32: 7.5% recall) nor by token id affinity (10%); by construction the
  predictable experts are already bank hits. Prefetch cannot help.
- More draft rows do not pay: DS4_V41_DSPARK_MIN_SURVIVAL=0.4 -> 4.0 rows/cycle, 2.62 accepted, but misses 1.7/layer
  and verify 145 ms -> 17-19 t/s (0.6: 3.0 rows, 2.16 accepted, 97 ms).
- Bank size: pct 40 (45 GiB) misses 1.65/layer, 17 t/s; pct 50 0.87, 18.3; pct 58 0.59, 19.6.
- GPU: 585 GB/s measured peak (compute read). Bytes per 3-row cycle ~13 GB (dense 7.2, experts 5.3) = 22 ms at peak
  vs 55-59 ms GPU span: kernels run at ~40% of bandwidth overall; ~10 ms/cycle of that is tiny kernels (bf16 rounding
  900/cycle, copies 255/cycle, norms, hc). That, and the vocab head + Markov loop in the draft, is what is left to
  chase on this architecture; the miss floor and the per-layer router sync bound the rest.
- Ceiling estimate at 16k, full quality, one node: GPU 55 -> ~35 ms with kernel work, misses ~25 ms at pct 58,
  turnaround 8, draft ~5 => ~75 ms per ~2.4 tokens => ~30 t/s. 50 t/s needs the experts resident (two nodes) or a
  different accept rate (the drafter is Q2-mismatched: 2.2-2.4 tokens/cycle).
- Later the same night: seed hotness 1..8 instead of 1..32 (9b1ec1c; warm-up misses 1.94 -> 1.70/layer) and the
  Markov walk chained on the GPU (1b28c2b; draft loop 1.9 -> 1.4 ms, the rest is the 132 MiB F32 W2 read per row --
  F16 W2 would halve it but changes the drafts). Same prompt: 18.7 t/s overall at pct 50, byte-identical.
- Also measured: plain (non-DSpark) decode on the 5.7k prompt 10.0 t/s; a short prompt with DSpark 19.5 t/s overall,
  22.5 steady (no seed for token-major prompts, so the bank fills from empty during the first ~70 cycles).
- Without the Markov head acceptance drops to 1.66 tokens/cycle (2.16 with): keep it.
- Left on the table, each a few percent, all needing the byte-identical gate: (1) bf16 rounding folded into the
  producers (19 dispatches/layer at ~4 us each measured for dependent tiny dispatches); (2) the router's 11 small
  kernels as one; (3) the hc F16 matvecs (68 outputs, K=7168) dispatch 3 threadgroups with nxpsg=8 -- nxpsg=32 would be
  ~10x faster but changes the summation order; (4) the 5 attention publish copies per layer merged; (5) a chunked
  Markov walk that stops at the survival cutoff instead of drafting all 5 rows.

## Sep 15 — 1M context (Neil: "the real target is 1M context, 16k is practically useless")
Setup: ds4-server on port 8010 with --kv-disk-dir so a long prefill is paid once (a 125k checkpoint is 0.8 GB and
loads in 0.2 s; continued/shutdown saves of longer sessions were not reusable as prompt prefixes because the stored
text includes the generated THINKING tokens). Prompts: ds4.c heads, 125k and 394k tokens. Prefill runs 540-600 t/s
at both lengths (encoder-first sweeps of 8192 rows; layers 0-19 over the whole prompt, then the decoder).
- Third panic: at -c 1048576 the context buffers are 25 GiB (block_mask alone is prefill_cap x ctx/8 = 4.3 GiB) and
  the bank budget reserved only dense+context, not the support model, the OS or other apps: anonymous memory went to
  150 GB (compressor 50 GB, swap 100 GB) within 90 s. Fixed in 409f569: reserve support model + 24 GiB headroom
  (DS4_SSD_CACHE_HEADROOM_GIB), pct default 70 of the rest (= the old 50 at 16k; 45-48 GiB bank at 1M).
- The 1M decode wall was the DSA indexer: the decode kernels (kernel_glm_indexer_scores_batch for verify rows,
  kernel_glm_indexer_score_one_direct for single rows) run one threadgroup per compressed row and re-read the token's
  16 KiB indexer q per row. 125k ctx: 6 ms per call x 8 index layers = 49 ms of a 110 ms sweep; linear in context,
  so ~400 ms per step at 1M (2-3 t/s). Routing decode through the tiled kernel (409f569): 3.5 ms per call at 394k.
  Then a decode-specific kernel (kernel_dsv41_indexer_scores_decode): q for the token in threadgroup memory, each
  simdgroup streams its rows once, 8 rows per pass so the simd_sum chains overlap: 2.2 ms per call at 394k with
  2-row passes, see below for 8-row. A first attempt holding K^T as simdgroup-matrix fragments across the head loop
  spilled and ran 20 ms per call. Output identical in every variant (555-char thinking prefix at 394k, 400 tokens at
  16k byte-identical).
- Decode at 394k after the indexer fix: verify ~115-125 ms (16k: ~97), 11.5-13 t/s vs 5-6 before; remaining
  context-proportional kernels: the indexer (18 ms/cycle), argsort causal shuffle 1.7 ms, indexed attention 2.6 ms.
- The prefill tail after a cache hit is slow: 461 rows took 32 s at 125k, 1060 rows 26 s at 394k. The rows go
  through the layer sweep in 32-row index batches (packed scores 2.7 ms + causal argsort 3.4 ms per batch per index
  layer) plus the decoder-suffix rebuild; ~20 ms per row against 1.7 ms per row for the 8192-row sweeps. Not fixed.
- Full-length run, 869k tokens of ds4.c (the whole file), server + disk checkpoints, DSpark, headroom 32 GiB:
  prefill 529 t/s average over 744k new tokens (23.5 min), decode 10.7 t/s over 120 tokens (11-15 steady, verify
  130-180 ms, 1.3 misses/layer with a 42 GiB bank of 4521 entries), wired peak 84 GiB, swap flat. Coherent output.
  Before this session's fixes the same run would have been ~2-3 t/s (indexer) or a panic (bank budget).
- What bounds 1M decode now: the indexer at ~3-4 ms per call x 8 (kernel is ~10x off its instruction budget --
  register-heavy transpose-reduce, worth a second pass), misses at 1.3/layer because 25 GiB of context buffers come
  out of the bank (block_mask alone is prefill_cap x ctx/8 *floats* = 4.3 GiB; as bits it would be 0.5 GiB), and
  the same per-layer sync and GPU inefficiency as at 16k. A tail sweep after a cache hit costs 15-30 s at any long
  context: whole-model read (~12 s at 9 GB/s) + residency commits for the seed (~2 s) + GPU idle during page-in.

## Sep 15 — prefill rows handed back through decode (40c5172 + follow-up)
- Of the 19.4 GiB of V4.1 context buffers at 1M, the batch rows (8192 x every per-row column, block_mask alone
  4 GiB) and the carry rows (3 GiB) are written only by a prompt sweep; through decode they are idle, and the bank
  cap left room for them. Measured first what the OS will take back: madvise(MADV_FREE_REUSABLE) on the memory
  Metal's own allocator hands out does nothing (footprint unchanged); on an anonymous mmap wrapped with
  newBufferWithBytesNoCopy it drops the footprint at once, contents preserved until the OS needs the pages;
  setPurgeableState:Empty also works but only for a whole buffer. Shared Metal buffers are wired only while a
  command buffer that uses them runs (wired rises by the buffer size during the blit and falls ~2 s after).
- Batch and carry tensors now come from ds4_gpu_tensor_alloc_reusable (mmap + bytesNoCopy; other backends alias
  ds4_gpu_tensor_alloc). After each prefill ds41_graph_release_prefill_rows advises rows 64.. of every batch column
  and all carry rows reusable; the next sweep that reaches them advises REUSE first (the drafter's verify sweeps
  only ever touch rows < 16). Reclaimed pages refault zero-filled; every such row is rewritten before it is read.
- The no-copy bank cap is recomputed from the requested budget with the released bytes taken out of the context
  reserve (ds4_gpu_stream_expert_cache_nocopy_recap): raised after prefill, restored -- and pruned -- before the next
  prefill, but only by the rows that prefill will sweep (ds41_graph_released_beyond), so a 461-row tail after a
  cache hit gives back 0.5 GiB, not 10. Numbers: 16k releases 5.3 GiB (cap 6050 -> 6452); 1M releases 10.4 GiB
  (batch 8.6 + carry 1.8; cap 5126 -> 5910 entries, 47.5 -> 54.8 GiB).
- Output check: the 16k reference (run_pin4_pct50) is reproduced byte-for-byte only with DS4_METAL_V41_INDEX_SCALAR=1,
  because 409f569 moved the drafter's 3-15 row verify batches onto the tiled indexer kernel (different summation
  order); the env now covers the Metal side too. With it set, the release build is byte-identical to the reference;
  without it, HEAD before and after this change produce the same (different) text.
- At 125k the bank was already big enough (0.00 misses/layer in the second half of 200 tokens; 3.0 ms per layer =
  1.95 GPU + 0.5 prepare + 0.5 turnaround), so decode is unchanged there (17-18 t/s steady). The first 50 tokens of
  every request run at 9.5 t/s regardless of bank warmth -- not misses; unexplained, worth a look (defer warm-up?).
- 869k decode anatomy (batch prof, second half of 120 tokens, bank 5910 entries): 0.00 misses/layer, so the
  bigger bank has removed misses at 1M entirely. Per layer (drain+prepare+turnaround): non-index layers 2.2-2.6 ms
  (the 16k floor), index-source layers 5.1-7.8 ms, layer 0 17.9 ms (carries the head, sampling and the 7 ms draft
  between cycles). Verify ~115 ms at n=2, ~175 ms at n=3; the 8 index layers' extra ~4 ms each is the indexer
  kernel (~0.8 ms/token) plus the top-k sort.

## Sep 15 — rewind instead of rebuild (e8085f3)
- Any prompt that was not an extension of the live tokens rebuilt the whole prefix (25 min at 869k): the server's
  rewind path was GLM-only because V4.1's compressors "cannot be rolled back by truncating row counts". They can,
  to a pair boundary: the caches are addressed by valid length and the odd-row pair state is rewritten by the next
  even token. What was really lost was the 128-row raw decode window. Every raw row now also goes to a per-layer
  ring of the last 8192 positions (DS4_V41_RAW_LOG_ROWS, 640 MiB), ds41_graph_rewind rebuilds the windows from
  it, and the server rewinds below a diverging prompt too (pair-aligned). tests/test_metal_rewind.c: a rewound and
  replayed session matches a session that reached the same tokens through a fresh prefix and the same replay with
  identical logits (a single sweep over the whole prompt computes the prefix rows in a different batch shape and
  lands up to ~1 logit away; the greedy tokens agree). Regenerate / edit-last-message at 1M now costs the tail.
- Caveat: rewinds deeper than 8192 - 128 tokens, or past a snapshot load, still rebuild.

## Sep 15 — top-k by radix select
- The indexer top-k (kernel_argsort_f32_i32_desc + merge) keeps every 1024-row block's top 512 and merges them in
  log2(blocks) passes: at 435k rows that is 9 dependent passes over 217k survivors, ~2-3 ms per call; the 32-row
  causal batches of prefill took 9.6 ms. kernel_topk_select_*: four 8-bit histogram levels find the k-th key
  exactly, one compaction pass, one threadgroup orders the winners (score desc, index asc). Same sets on random
  rows (tests/test_metal_topk_select.m); only when more than 2048 rows tie at the boundary is the choice among
  them arrival-ordered. First timings with the GPU shared: 435k/3 tokens 3.0 -> 0.35 ms, 869k 2.7 -> 0.43 ms,
  32-row causal batch 9.6 -> 0.97 ms. DS4_METAL_DISABLE_TOPK_SELECT=1 restores the sort.

## Sep 15 — indexer decode kernel as simdgroup matrix products
- kernel_dsv41_indexer_scores_decode now forms C(8 rows x 32 heads) = K(8 x 128) . Q^T(128 x 32) from 8x8
  simdgroup_float8x8 tiles: 16 key tiles read once from the cache and 64 q tiles from threadgroup memory per 8
  rows, then a per-row relu/weight reduction. The row-streaming kernel spent ~280 instructions per row and lane
  (32 float4 dots, a 31-shuffle transpose-reduce and its selects); this one ~25. Standalone at 435k rows with the
  GPU shared by a running prefill: 3.6-4.5 -> 0.59-0.82 ms per token; |diff| vs a double reference 8e-10.
  DS4_METAL_V41_INDEX_DECODE_ROWS=1 keeps the old kernel.
- Early bank seed (opt-in, DS4_METAL_STREAMING_PREFILL_EARLY_SEED=1): a fresh 869k prefill (empty bank) ran
  20-25% slower than the same prefill over a bank a previous session had filled, so the encoder sweeps can seed
  layers 0-19 while the bank is below a quarter of its cap. A/B on the 125k prompt: 296 s with, 339 s without,
  but the first sweep alone swung 435-720 t/s between identical runs -- prefill I/O depends on what the page
  cache still holds of the model from the previous run -- so the default stays off until measured on a quiet
  machine. The topk select did show in the same runs: first sweeps of 713-727 t/s against 672-675 before.
- Open: the server's cold/continued disk stores of a raw 1M prompt (570401 tokens, "key=token-text") are not found
  again as a text prefix of the same prompt, so the 25-minute prefill repeats after every restart; only the store
  of the exact p100k prompt hits. Not investigated. Watchdog note: watch3.sh's compressor limit was raised 25 -> 40
  GiB after it killed a run at comp=26G with swap flat (wired 79G) -- the 1M graph's idle buffers get compressed.

## Sep 15 — 869k decode timeline (GPU shared with a running Synology backup; relative shares only)
- Encoder timeline over 300 tokens at 869k (kernel sum 143 ms of a 163 ms GPU span per verify cycle; ~227 ms
  wall per cycle, so ~60 ms/cycle is CPU-side: per-layer turnaround 0.5 ms + prepare 0.5 ms x 40, the 7 ms
  draft, the head):
  - 23.4 ms  kernel_dsv41_indexer_scores_decode (the new MMA kernel), 8 calls of ~2.9 ms for 2-3 tokens over
    435k/869k rows: every token re-reads the key cache (222-445 MB per call per token), so it sits near the
    per-token bandwidth floor. Reading K once per cycle needs q for 2-3 tokens in threadgroup memory, which at
    16 KiB/token fp32 does not fit beside the C tiles; an f16 index cache (the GLM path has cache_f16 already)
    would halve both the traffic and the q footprint. Next step if the indexer is worth another ~12 ms/cycle.
  - 34.4 ms  routed expert matvecs (iq2_xxs pair_swiglu 21.1 + q2_K sum6 13.3): ~12 unique experts x 9.5 MiB
    per layer in 0.86 ms = ~23% of peak bandwidth. The largest single item and the same inefficiency as at 16k.
  - ~40 ms   dense/attention matvecs (q8_0 r1_2 11.8, attn_out_low 9.1, q8_0 r1_3 6.1, f16 9.2, f32 4.0, ...).
  - 7.6 ms   kernel_cpy_contig_u32_4, 291 copies per cycle (7 per layer: window, rewind ring, pair state,
    cache rows); 4.9 ms kernel_dsv41_bf16_linear, 871 per cycle (22 per layer, 5.6 us each); 1.8 ms rms_norm.
    ~14 ms/cycle of tiny dispatches that could fold into their producers.
  - 5.5 ms   indexed attention; the top-k select is now invisible (its kernels are below the 16-line cut).
- Prefill tail of the same run (last 3000 command buffers): kernel_dsv41_indexer_scores_packed 8.2 ms per
  call (2404 ms of 7395), the mixed attention dual 30 ms per call, expert mm_id 15 ms per call;
  kernel_topk_select_hist 58 us x 7348 = 428 ms (was the causal argsort at 3.4 ms per batch).
- The server's cold disk store of the 1M prompt now hits (c5ce06b): the second 869k prefill of the day went
  570401 -> 868904 in 13.7 min instead of the whole prompt in 25-38.

## Sep 15 — deferred verify batches (no per-layer drain in the DSpark verify sweep)
- A verify batch of up to 32 rows now takes the deferred route the one-row decoder takes: the address kernels
  read the router's output and the per-layer GPU tables that install/evict keep current, the residency set
  covers the bank, and the miss flag is read once at the end of the sweep. A batch that missed is rolled back
  (`ds41_verify_rollback`, which now also repairs the 128-row windows from the rewind ring — see below) and
  rerun with the stops in place, so nothing wrong is ever committed. The batch is not held to the one-row
  warm streak (the drafter's own installs never let it reach 4); the backoff after a real miss still applies.
- Two bugs found on the way, both pre-existing in spirit:
  - `ds4_gpu_routed_moe_batch_tensor` never cleared `g_stream_expert_defer_dispatch` at entry (the one-row
    paths do). Once a deferred batch set it, every later batch — the stopping rerun of a batch that missed,
    and each verify batch after it — substituted the deferred entry list for the resources it had just
    prepared: its experts were never marked in flight, `prune_global` could evict them under the running
    dispatch, and the address kernels then skipped those experts without a stop to catch it. Symptom: after
    the first deferred miss, "sweep, roll back, sweep" stopped being exact (logits off by up to 13, exactly
    reproducible, KV state verified intact by hashing every persistent buffer across the rollback).
  - A DSpark verify sweep of n rows lands in the window slots of positions [start-128, start-128+n) — the
    oldest rows a token at `pos` still attends to — and a partial commit or rollback never restored them.
    `ds41_verify_window_repair` (from the rewind ring) fixes it; the 16k reference outputs before this fix
    were subtly wrong (they no longer match `run16k_old.txt.gen`, and should not).
  - Also: a layer whose every selected expert is served from overflow views refused to encode (the in-flight
    marker rejects an empty list); only reachable with a bank small enough to hold none of the layer.
- Verification: `DS4_METAL_V41_BATCH_DEFER_CHECK=1` reruns every deferred batch with the stops and prints the
  worst logit difference — 0 for every batch at 16k; 16k DSpark output with deferral is byte-identical to
  `DS4_METAL_V41_DISABLE_BATCH_DEFER=1`. `DS4_METAL_V41_DEFER_DEBUG=1` prints the layer-0 gate decision.
- Testing hygiene after the 14:33 hard lock: the 16k debug runs now use `DS4_SSD_CACHE_HEADROOM_GIB=52`
  (a 36 GiB bank instead of 55; also exercises the miss path), and the watchdogs kill at wired>92 GiB /
  compressor>20 GiB / swap>3 GB (ds4) and 96/24/3 (ds4-server). The lock came 6 minutes after a run that
  had held wired at 87 GiB with the compressor at 21 GiB and ~60 MB free for its whole duration.
- Loop rule (Neil): `git fetch` before every commit; if origin/main moved, rebase, rebuild, rerun the 16k
  byte-identity check, then commit. As of 15:30 the branch is 0 behind / 57 ahead of antirez/ds4 main.

## Sep 15 — antirez's regression sweep, branch vs main (ds4-bench, 2k→64k step 2k, 128 greedy tokens, no DSpark)
- Same machine, same 23 GiB bank (`--ssd-streaming-cache-experts 30GB`, no-copy), backup off, Cursor/Codex/
  Chrome/Hermes resident (~40 GB of other apps):
  - generation: branch 14.7 t/s mean vs main 11.4 (+29%, every frontier; 49k: 17.3 vs 12.3).
  - incremental prefill (each 2k chunk after 128 decode tokens): branch 76.6 t/s vs main 111.3 (−31%);
    first token after prefill 330 ms vs 150.
- The prefill loss is the bank pinned through the sweep (686bd53 publishes the residency set at seed time and
  the set stays on the queue through decode): the per-layer expert reads (`map`) took 10–12 s per 2k chunk
  against 7 s on main and 4–5 s with the bank unpinned (the reads come from the file cache when it has the
  room; wired memory during the sweep was 60–76 GiB pinned vs 24–29 unpinned). Not the released prefill rows
  (DS4_METAL_DISABLE_REUSABLE_TENSORS=1 changes nothing), not Engram.
- Unpinning for short sweeps (`DS4_METAL_STREAM_EXPERT_PIN_PREFILL_TOKENS=32768`) gets prefill to 128–131 t/s
  (above main) but decode then pages the bank back in: 3.7 s before the first token and ~8 t/s over the next
  128 instead of ~15, and every later layer of those tokens runs with the stops because the one publish a
  token is allowed goes at layer 0 while the installs come after it. Default stays pinned (threshold 0); the
  knob is there for prefill-dominated workloads. A 2k+128 turn is a wash either way; longer generations
  favour the pin, very short prefills favour it too.
- Answered: wired memory grows 1:1 with the bank, not 2x (plain 16k decode, bank 25.4 / 41.5 / 56.9 GiB ->
  wired 62 / 76 / 86 GiB average over the last 30 s; the 40 GiB delta in the sweep was the bank plus the
  prefill reserve and the rest of the session). Same runs: plain decode 200 tokens after a cold 10k prefill is
  7.4-7.7 t/s whatever the bank size -- the first couple of hundred tokens are the warm-up (installs, no
  deferral), and the 14-15 t/s of the sweep is what a warm bank does (its frontiers accumulate 128 tokens each).
- Per-feature ablation by ds4-bench (fresh 16k/48k prefill + 128 tokens) cannot see the decode features for
  the same reason: every run is in warm-up. Standalone numbers on a quiet GPU instead: top-k select vs sort
  65k 0.89 -> 0.37 ms, 435k 2.36 -> 0.40, 869k 3.34 -> 0.41, 32-row causal 4.94 -> 1.82 (4096 rows: 0.83 vs
  0.71, k=1 at 129k: the sort wins 0.43 vs 0.85); indexer rows kernel vs MMA 435k x 3: 0.72 -> 0.50 ms/token,
  869k: 1.45 -> 1.09 (the MMA kernel reads K at 445 GB/s, i.e. at the floor -- the earlier 3.6 -> 0.6 was
  measured with the GPU shared, which hurt the old kernel more).

## Sep 15 evening — clean 869k decode with the deferred batches in (backup off)
- Server, 570k cold store hit -> 298503-token prefill in 562 s (531 t/s avg) -> 300 tokens: 10.2 t/s on the
  first request, 8.7 / 8.0 / 8.7 on rewind-based repeats (swap had reached 0.8 GB; Cursor/Codex/Chrome
  resident). Morning baseline under the backup: 10.7. No gain at 869k from the day's decode work.
- Why: the bank at 1M is capped at 5078 entries (47 GiB) = ~127 of 256 experts per layer, and a verify batch
  of ~4 rows routes to ~20 unique experts per layer, so a deferred batch misses with near certainty: 7 of 698
  verify batches ran deferred, all 7 missed and were rerun with the stops, and the backoff kept the rest
  stopping. Batched deferral needs a bank that holds nearly every expert the batch can route to -- true at 16k
  with a 55 GiB bank and a warm LFU, not at 1M on a 128 GiB machine with 40 GB of other work resident.
- Timeline (611 cycles, 1.73 tokens committed per cycle): GPU span 161 ms/cycle, kernel sum 143 ms, wall ~226
  ms -> ~65 ms/cycle CPU-side and gaps. Kernels: indexer 21.9 ms (8.1 calls x 2.7 ms, i.e. the quiet-GPU MMA
  rate), expert matvecs 33.1 (pair 20.4 + sum6 12.7), dense q8_0/f16/f32 matvecs ~40, attn_out_low 8.6,
  copies 7.9 (320/cycle), bf16_linear 5.5 (875/cycle), indexed attention 4.9, rms_norm 1.8.
- Where 50 t/s stands against physics: the dense weights are ~9 GiB per token at ~500 GB/s = ~19 ms of pure
  reads per cycle, shared by the 1.7 rows of the cycle; experts ~4.5 GB per cycle = ~9 ms; the indexer's K read
  ~11 ms at f16. With every kernel at the bandwidth floor and the host overhead gone, a cycle is ~55-60 ms for
  ~1.7 tokens, i.e. ~30 t/s. 50 t/s at 1M needs that AND a DSpark that commits ~3 tokens per cycle (block 5,
  currently 1.7 agreed). The gaps today: dense kernels at ~45% of bandwidth, expert kernels at ~27%, the
  indexer re-reading K per token, ~15 ms of tiny dispatches, ~65 ms host-side.

## Sep 15 evening — where the host-side time of a verify sweep goes (16k, 36 GiB bank, stops)
- `DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_PROFILE=1` now splits the per-layer preparation:
  hot/loop/res, loads (no-copy installs), wrap (3 views per install), ra (3 F_RDADVISE per install), install.
  Per routed layer of a 3.25-row batch: 14.6 unique experts, 3.42 installs, prepare 1.3-1.5 ms of which
  read-ahead 1.35 ms (0.4 ms per install), the three view creations 0.07 ms per install, the rest ~0.
  The read-ahead is real I/O issue, not waste: with it disabled prepare drops to 0.36 ms but the GPU faults
  the pages in serially and the drain goes 2.7 -> 7.1 ms per layer (sweep 206 -> 400 ms, 10.7 -> 5.8 t/s).
  A mincore gate before the advise made it worse (0.25 ms per mincore call, and the pages are not in core).
- So at 16k with a 36 GiB bank every verify sweep still reads ~1.3 GB of experts (3.4 misses x 9.5 MiB x 40
  layers); the same run with a 55 GiB bank earlier in the day had ~0 misses. The bank size is the whole game
  for misses, and 36 -> 55 GiB is the difference between 23% and ~0% miss rate per layer for 3-row batches.
- The other host cost, ~0.75 ms per layer between drains, is the encode of the layer's ~50 dispatches in
  ds4.c (~30 ms per sweep) -- the same tiny dispatches that cost ~15 ms of GPU time. Fusing bf16 rounding and
  the copies into their producers attacks both.
- gentext.py's hash column is Python's per-process string hash (useless); the `cmp` of the .gen files is
  what the identity checks used, and prof5/prof7/prof8 (read-ahead on/off) are byte-identical.

## PR series (local topic branches on antirez/ds4 main 9139e2a, in worktree ../ds4-pr)
1. pr/server-disk-store-rendered-lookup (2d206c2): the KV disk-store lookup fix + tok_roundtrip. `--server` OK.
2. pr/indexer-topk-select (a372d4c): radix-select top-k + test. Byte-identical at 16k, `--metal-kernels` OK.
3. pr/indexer-decode-kernels (6856a23): 409f569's indexer hunks + 90a7561 + the MMA kernel; identical output
   at 16k across MMA/rows/one-row, scores agree to 5e-7, `--metal-kernels` OK.
Next: prefill rows release/reuse, rewind; the RFC text is in bench/pr/04-rfc-streaming-decode.md.
Not pushed: waiting for Neil to say the fork under gaineyllc is fine.

## Next levers at 1M, in order of expected gain (written Sep 15 19:20)
1. DSpark acceptance. 1.73 tokens committed per cycle at block 5 is the multiplier on everything below: a
   cycle's dense reads (~19 ms floor, ~40 ms today) are shared by however many rows the batch has, so extra
   rows are cheap at 1M and longer blocks alone do not help (the marginal row is ~30 ms of expert reads and
   indexer, and adds ~0.25 accepted tokens). What helps is accepting more of the same rows: tree drafts
   (top-2 alternatives at the first uncertain positions, verified as extra rows at the same position; the
   verifier commits a path instead of a prefix). Needs per-row attention over prefix+own path and a path
   commit in the partial-commit code (ds4.c ~84720). Expected +30-50% on committed tokens per cycle.
2. Expert matvecs (33 ms/cycle at ~27% of bandwidth): iq2_xxs pair_swiglu 510 us and q2_K sum6 317 us per
   layer for ~12 unique experts x 9.5 MiB = 114 MiB -> 137 GB/s. Issue-bound (dequant tables); the table-
   driven variants tried Sep 15 were slower. Next: process 2 rows per weight read where the batch has them
   (the pair kernel reads each expert once per row today), and a q2_K sum6 that keeps the 6 experts' partial
   sums in registers across the row loop.
3. bf16 rounding + copies + norms as producer epilogues: 875 + 320 + 194 dispatches per cycle, ~15 ms GPU and
   ~15 ms host. The mul_mv_ext family stores at one site (dense.metal kernel_mul_mv_ext_q4_f32_impl, the
   `dst_f32[i01] = sumf[ir1]` line): a `round_bf16` flag in ds4_metal_args_mul_mv (trailing field, zero in
   the positional initialisers of moe.metal) and in ds4_gpu_mul_mv_ext_args / ds4_gpu_q8_0_matvec_args on the
   host, plumbed through ds41_matmul_batch's bf16 argument to ds4_gpu_matmul_q8_0_* and the projection_rows
   paths; the same flag in kernel_rms_norm_mul_f32_4 for ds41_norm_batch. Byte-identity must hold (the
   rounding is bit-exact wherever it is applied). The copies: let ds41_attention_batch's attention read the
   window ring in place instead of staging through raw_prefill (3 of the 7 copies per layer).
4. Indexer K read once per cycle for the 2-3 tokens (22 -> ~11 ms): f16 index cache (the GLM path has
   cache_f16) and q for the batch's tokens in threadgroup memory.
5. The host side (~65 ms/cycle at 869k): mostly the read-ahead of real misses (bank at 50% of the experts)
   plus ~30 ms of encoding; (3) halves the encoding; the misses only shrink with RAM.

## Sep 15 night — the expert kernel is 3-4x slower in the engine than standalone
- ~/ds4-bench/mb/pairbench.m compiles ds4's own metal sources and runs kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32
  on synthetic experts (2048 rows x 4096, iq2_xxs, 12 distinct experts per layer, 10 layers' worth so nothing is
  cache-resident): 0.131 ms per layer for 2 tokens = 396 GB/s -- at the bandwidth floor. The same with no-copy
  views over the real model file wired by a residency set: 0.131 ms. In the engine the same kernel, same shape
  (2-4 tokens x 6), takes 0.41-0.55 ms; DS4_METAL_PAIR_TWICE=1 dispatches it twice on the same experts and the
  second is as slow as the first (550/563 us), so it is not the memory source or page residency.
- Idle gaps: with usleep between single-layer command buffers the harness's best time goes 0.19 (no gap) ->
  0.24 (1.5 ms) -> 0.44 ms (3 ms) for 3 tokens: the governor drops the clock in the host's per-layer gap. A
  spinning keep-alive thread (DS4_METAL_DECODE_KEEPALIVE=1, the TP one) makes decode worse (8.3 vs 10.2 t/s:
  it competes). A short spin submitted only at each drain (DS4_METAL_DECODE_KEEPALIVE=2, 150k iters ~1 ms) is
  +3-5% at 16k over 3 pairs of runs (10.7-11.0 vs 10.3-10.6 t/s, sweep 177-182 vs 184-195 ms), byte-identical.
  Left opt-in. So clocks are part of it but not the 3-4x.
- Still unexplained: the rest of the gap. Candidates: the read-ahead's SSD DMA into the page cache during the
  sweep (1.3 GB per sweep at 16k with a 36 GiB bank, 2.8 installs/layer even at 51 GiB in a fresh process),
  and the 7000-allocation residency set. The separating test is a long warm session (0 installs/layer) with
  the timeline gate: if the pair kernel is then ~0.15 ms, the miss I/O is what slows every kernel, and the
  1M budget math changes (expert reads at the floor would take 9 ms per cycle, not 33).

## Sep 16 early — what the in-engine slowdown is and is not
- Ten back-to-back dispatches of the pair kernel on the same experts inside the sweep (DS4_METAL_PAIR_TWICE=10):
  all ten equal, 315-390 us for ~19 (row, expert) pairs = 16-20 us per pair, against 11-12 us in the harness
  (ne01 2304 corrected). So the steady in-engine cost is ~1.5x the harness, not 3.5x; the rest of the 510-550 us
  a normal sweep pays is the first dispatch after the host gap (the second half of a warm run is 315 us). Clock
  and pipeline warm-up per layer, i.e. the per-layer stop, is the main multiplier on the expert kernel time.
- MAP_SHARED vs MAP_PRIVATE for the model file: no difference in the harness (0.131 ms either way). 8k extra
  resident views: no difference.
- Async read-ahead on a helper thread: prepare 1.3 -> 0.3 ms per layer but the drain grows 2.2 -> 3.2 ms; the
  GPU waits for the same pages. Net zero (10.0/9.4 vs 10.2/10.0 t/s). Removed. Miss I/O is on the critical path
  whoever issues it; only a bank with fewer misses removes it.
- Warm 16k DSpark with a 47 GiB bank over 800-2500 tokens: 13.9-16.1 t/s (the 40-token runs' 10.5 is warm-up);
  installs never reach 0 at this bank size with 3-4-row batches (1.2-2.2 per layer).

## Sep 16 — 256k baseline and what 50 t/s there requires
- Server, 261307-token prompt (first 822k chars of p1m.txt), `-c 294912`, bank capped at 48 GiB (58 GiB got the
  process killed at 3.4 GB of swap with the other apps resident), DSpark: first 500 tokens 14.7 t/s, clean
  repeats 12.1 / 14.6, a 2000-token run 15.3 t/s steady (819 cycles, 2.44 tokens per cycle, verify 161 ms per
  cycle). Deferred batches: 8 attempted over ~2600, all 8 missed. Misses are uniform across layers (1.1-2.0
  installs per layer per batch even after 2500 warm tokens at 16k with 47 GiB), because a 3-row batch routes to
  ~14 unique experts per layer and the bank holds ~130 of 256: P(no miss) ~ 0.9^14 = 23%, and a miss costs a
  full rerun. Partial rerun from the missed layer would not help (the first miss is at layer 0-1).
- The floor at 256k with no stops: dense ~19 ms + experts ~8 + indexer ~4 + attention ~5 + tiny ~5 = ~45 ms per
  cycle for 2.44 tokens = ~53 t/s. The stops (host prepare 1.3 ms + GPU pipeline/clock warm-up after each of
  40 gaps, kernels at half speed) are the entire distance from 15 to 50. They are forced by misses, and the
  misses by memory: all 10240 Q2 experts are 97 GB. On this 128 GB machine with ~40 GB of other work resident
  the bank tops out near 48-58 GiB. A dedicated 128 GB box gets ~90 GB (still short); 192 GB, or two machines
  (ds4's TP/pipeline path, each holding half the experts), gets every expert resident, and then the deferred
  batches -- already exact -- take the stops out.
- Instrumentation cost: the encoder timeline itself slows decode ~30% (8.4-9.7 vs 12-15 t/s); never measure
  t/s with it on.
- Server gotcha: prompt + max_tokens + prefill_cap (8192) must fit -c; a 2000-token request at 261307 with
  -c 270336 silently re-synced the session (a full 261k prefill) mid-generation.

## Sep 16 — split-miss dispatch (DS4_METAL_V41_SPLIT_MISS=1), and what the machine's memory really holds
- Built: prepare leaves the missing experts pending, the gate/up pass runs for the resident ones and is
  committed, the misses are installed while it runs, a second gate/up pass covers only the pending slots
  (resident slot ids set to -1), then the one down pass. Exact (4/4 identical), +2% at 16k (10.4/10.0 vs
  10.2/9.9 t/s): the layer still waits on the miss I/O, and ~0.5 ms of GPU work is all that hides behind it.
  Opt-in; not the answer.
- With ds4 idle: all processes' resident sets sum to 14.6 GB (the earlier "~40 GB of other apps" was wrong:
  that was ds4's own bank plus 16 GB of compressor holding the apps' pages my runs had squeezed out). A booted
  iOS simulator, Hermes, ChatGPT/Codex, Cursor, VS Code and Chrome are the tenants. With them quit and
  iogpu.wired_limit_mb raised, the bank should reach ~85-90 GB at 256k (experts total 97 GB): the coldest
  ~10% absent, which is the regime where a 3-row batch's 14 unique experts are usually all resident.

## Sep 16 — the memory experiment at 256k (apps quit, wired limit 115 GB)
- Bank 48 GiB -> 15.3 t/s; 65 GiB (cap from headroom 6) -> 17.1; 79 GiB (`--ssd-streaming-cache-experts 92GB`,
  capped to 86 by the working-set budget; 8509 entries = 83% of the experts, wired 91-92 GB) -> 17.2-18.8 avg,
  chunks to 24 t/s. Deferred batches with the backoff capped at 8 (DS4_METAL_V41_DEFER_BACKOFF_MAX): 209
  attempts warm, 209 misses -- even 2-row batches, even at 83% residency. The absent 17% are not "cold": the
  router reaches them in every batch. Deferral needs ~100% residency (97 GB of experts + ~30 GB of everything
  else), which this 128 GB machine cannot hold; the driver's working-set budget also stops the bank at 86 GiB.
- So on this machine 256k decode is ~17-19 t/s steady (24 peak), 1M ~10, and the route to 50 at 256k is
  memory: 192 GB, or two machines splitting the experts (ds4's TP/pipeline path). Software-only remaining
  levers here are single digits each (keep-alive +3-5%, split-miss +2%).

## Sep 16 — agent turns: short continuations sweep against the bank (commit "metal: short continuations
## sweep against the expert bank")
- The realistic workload (Hermes-style agent turns on a live 260k session) paid the whole layer map for
  every turn: 1080 rows at 261k = 42-48 s (97 GB.. see below, read end to end), and below 1024 rows the
  warm-append rule went token-major at 17 t/s. The store-hit tail (1211 rows) ran the map at ~23-28 s.
- Built: `g->bank_prefill` -- a tail of <= 4096 tokens on a warm session sweeps in map-sized chunks (2048)
  with the selected-address batch path instead of `metal_graph_stream_map_layer`. First cut used the
  per-row address pair kernel: 540 rows = 337 ms/layer, compute-bound (it dequantizes every expert once
  per row), 2x540 rows 36 s vs 42 s -- no. Second cut runs the map path's own grouped matmul (map0 +
  `kernel_mul_mm_id_iq2_xxs_cached_f32_mpp` gate/up + new `kernel_mul_mm_id_q2_K_cached_f16_mpp` down,
  EXPERT_ADDRESSES instantiations) through the bank's address table: 1080 rows in one chunk 18.9 s and
  byte-identical to the map (the 2x540 split was not identical: chunking changes the bits, so the bank
  path cuts exactly where the map cuts).
- Then the bank churn: a 2048-row chunk touches ~300 of 384 experts in every layer (12k entries, more
  than the 8509 the bank holds), and prepare's frequency hotness (rows per expert) made every prefill
  expert hotter than decode's set, so the sweep evicted the whole bank and the next chunk missed 72-76%
  again; decode after the turn fell to 2.4 t/s. Fix: entries a bank prefill installs are `bank_transient`
  (first victims regardless of hotness, cleared when decode hits them), and a bank prefill neither ages
  nor heats the hotness table; the keep-list seed still runs at the end. Miss rate 36% on every chunk,
  decode after the turn 7 -> 15 t/s (map: 5.4 -> 15).
- Numbers (M5 Max, 256k live session, temperature 0, identical md5 of the completion text in both modes):
    1080-token turn   map 42-48 s      bank 15.1-15.8 s (72-76 t/s)
    3770-token turn   map 74.9 s       bank 46.5 s (82 t/s; 2048 chunk at 100 t/s, 1722 at 68)
  Where the 15 s goes for 1080 rows: ~4k expert loads x 9.5 MiB = 37 GB of random reads (36% miss) at
  2.5-4 GB/s effective, overlapping only partly with ~4.4 s of MoE matmul + ~2.5 s of attention. The next
  lever is the miss rate, i.e. bank residency again.
- Correction to the memory notes above: this model has 384 routed experts per layer (gguf
  `deepseek41.n_routed_experts`), not 256 -- 15360 x 9.49 MiB = 142 GiB of experts, so the 8509-entry
  bank is 55% residency, not 83%; the "97 GB" figure was for 10240 experts. 100% residency needs ~142 GB
  of experts + ~30 GB: a 192 GB machine fits, two 128 GB machines splitting the experts fit.
- Upstream moved 20 commits (Qwen batched MTP, 400 files); the branch rebased clean onto 8db1d1d and the
  numbers reproduce on the rebased build.
- Follow-up (commit "a bank prefill sweeps the whole tail at once"): one sweep for the whole tail (<= 8192
  rows) instead of map-sized sweeps; the second 2048-row sub-chunk then misses 5% instead of 37%. Map vs
  bank, identical completions: 1080 rows 46.2 -> 17.0 s; 3770 rows 74.5 -> 31.7 s; 5534 rows 79.5 -> 39.9 s
  (142 t/s). The store-hit tail after a server start (empty bank) still takes the map, correctly.
- Next on this path: overlap the misses with the matmul (two passes per layer over the address table,
  resident experts first while the misses install -- the cached mm kernels already skip a zero address),
  worth ~30% of a turn; after that the turn is the SSD reading the 36% of experts the bank lacks.
- Split-miss for the bank sweep (resident experts' matmul first, misses installed meanwhile, second
  pass over the misses; exact): no change, dropped (stash). The read-ahead prepare issues already
  overlaps a layer's miss I/O with its dispatch; the MoE stage of a 1080-row turn is the SSD (37 GB
  of misses at ~4 GB/s), not the matmul. What did pay: issuing the F_RDADVISE calls from a helper
  thread (they cost ~0.25 ms each on the sweep's thread while the GPU idled): 1080 rows 22.1 -> 18.8 s,
  3770 rows 37.1 -> 31.9 s, same bits.
- PR #1060 (radix-select top-k) failed `make test-deepseek41-metal` at the 4096 frontier: the causal
  batch and the single-token call chose different algorithms for the same row, and ties past the
  2048-tie cap were arrival-ordered (nondeterministic). Fixed (a batch straddling 4096 is cut in two;
  a tie-count pass per chunk ranks ties by index with no cap), PR amended and its body rewritten to
  say plainly where the select and the sort differ (only at exact ties at the k-th boundary). Lesson:
  run test-deepseek41-metal, not just ds4_test --metal-kernels, before opening a PR.

## Sep 16 — Edge0 prerouter (Edge0-35B-A3B-preview, github.com/Edge0-AI/edge0, paper/main.pdf): what transfers
- Their mechanism: a per-layer head (fc1 -> erf-gelu -> fc2, plus a linear path warm-started from the
  NEXT layer's router weight) takes layer N's post-attention-norm hidden state at token t plus one-hots
  of the experts layer N routed to at t and t-1, and predicts layer N+1's routing at token t+1 (double
  shift: one layer, one token). One flush per step predicts every staged layer; the next step's SSD
  reads overlap the current forward. They tried same-token per-layer prediction (Pre-gated MoE style)
  and every variant lost to a plain LRU: the per-layer sync drains the GPU pipeline (30-100 ms/step),
  which is the same "stops" cost we measured. 33 fp16 heads, 0.2 GB. +59..84% decode in their regime.
- NOT transferable: "prediction is the routing". At decode they route by the head's logits, not the
  model's router, so the staged set is the routed set by definition ("zero drop"). Requires training
  (distilled heads + a recovery LoRA trained on the student path) and costs quality: 3.9 points mean
  on the 35B tier, 6.1 on AIME. That is a different model; ds4's bar is exactness vs the reference.
- NOT transferable: "35B in 3 GB". Qwen3.6 4-bit experts are ~7 MB per layer per token at K=4 (310 MB
  per layer); V4.1 Flash Q2 is 6 x 9.5 MiB x 40 = 2.3 GB per token, 30x more. Their engine reads nearly
  every routed expert from SSD every step (15-18 tok/s on M4 Pro; 19.9 with the prerouter OFF). On this
  model that is ~2 tok/s from the SSD alone, which is why the 79 GiB resident bank carries us.
- Transferable, exactness kept: the prediction as a PREFETCH HINT, the true router still decides.
  ds4 already has the naive form (ds4_gpu_stream_expert_predicted_begin_load: reload a layer's
  previous-token experts before it runs); Edge0 quantifies why it is weak: adjacent tokens agree on
  only ~a quarter of a layer's experts. A one-token-ahead head would raise the hit rate of the
  deferred verify batches (100% miss today at 55% residency) without changing any output.
  Step 1 (no training): their warm-start init is "apply layer N+1's router to layer N's hidden state".
  Instrument decode to compute that proxy per layer and log its top-6 recall against the real routing
  one token later; if recall is 60-80%, prefetch its top-8..12 per layer. Step 2 (training): port their
  scripts/ distillation to V4.1 Flash (384 experts, sigmoid/group routing like their 8B tier).
- Also worth taking regardless: pin_bonus -- experts a predictor names get a bonus in hot-set
  selection (our victim ranking), so what is about to be used is kept resident.
- Their other findings match ours: the floor is host-side per-step work (44 ms/step of graph building
  for them), and prerouter gains shrink on hot caches and fast storage.
- Probe result (DS4_V41_PREROUTE_PROBE=1, 10k prompt, 320 greedy tokens, no DSpark): the zero-training
  proxy "layer N+1's router applied to layer N's post-FFN-norm hidden state at token t" recalls
  22.6% of layer N+1's real top-6 at token t+1 (31.9% with its top-12); the same-layer previous-token
  set ds4 already prefetches recalls 25.9%. Per layer 7-36%, worst at 1-3, 15-16, 19, 39. So Edge0's
  warm-start init is no predictor on its own: their gain is the trained MLP correction (and the
  prev-token one-hots), which needs the distillation run. Nothing to ship from the probe; the code
  stays as a diagnostic (ds41_probe_* in ds4.c). Decision: a trained head is a separate project
  (port scripts/ from Edge0 to V4.1 Flash, 384 experts, sigmoid/group router; a data pass through
  the model on this machine); the cheap win to keep is pin_bonus once any predictor exists.

## Sep 16 — antirez's tracks at 256k (bench256.sh: ds4-bench --ctx-start 16384 --ctx-max 262144 --step-incr 16384
## --gen-tokens 128 --ssd-streaming, promessi_sposi.txt, NOCOPY=1 HEADROOM=2 PCT=100)
- Correctness track on the branch (streaming build, DS4_TEST_SSD_STREAMING=1, 60 GB bank): --server OK,
  --metal-kernels OK, --long-context OK (30474-token recall), --logprob-vectors ERR on one vector
  (short_code_completion step 0 selected token) -- identical failure on upstream main 8db1d1d with the
  same Q2 file, so it is the quant vs the official API vector, not the branch. make test-deepseek41-metal
  PASS, make test-metal-ssd-experts PASS, make test-metal-moe-prefill PASS.
- Speed track, upstream main 8db1d1d at --ssd-streaming-cache-experts 30GB (92GB does not fit main: it
  copies experts into the bank; 0.58 t/s decode at 16k with 6 GB swap): prefill 620 t/s at 16k falling
  to 393-416 at 229-262k; decode 11.8 at 16k, 10.1-10.6 from 98k on; wired flat at 49 GB.
  (bench_main30.csv). Past 229k ds4-bench replays the whole prefix per frontier (snapshot > 1 GiB).
- Speed track, branch at 30GB: FAILED at the 65536 frontier -- "Metal command batch failed: Insufficient
  Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)" in decode right after the prefill, wired
  63 -> 80 GB over the first frontiers where main stays at 49. First three frontiers: prefill 528/442/328
  t/s, decode 10.1/10.6/9.8 (bench_branch30.csv). Re-running with DS4_METAL_V41_BANK_PREFILL_MAX=0 to
  split today's bank-prefill path from the older bank-shrink/no-copy state (bench_branch30nb). Until
  the wired growth is found the branch does not pass the speed track; the server at 92GB, -c 294912,
  never showed it (different shape: 16k prefills per frontier under a 262k allocation).
- bench_branch30nb (same, DS4_METAL_V41_BANK_PREFILL_MAX=0): identical failure at the 65536 frontier, wired
  5 -> 56 GB inside the first 16k prefill and 72-81 GB by the third frontier. So the growth is the older
  branch state (no-copy bank / bank-shrink / prefill pins under the layer map), not today's bank-prefill
  path. First frontiers at 30GB are also no faster than main (prefill 543/445/378 vs 620/442/328;
  decode 10.1/10.4/10.0 vs 11.8/10.6/9.8): the branch's gains are bank-size dependent (92 GB) and the
  agent-turn path, which ds4-bench does not exercise.
- Where to look next (not done): what stays wired across frontiers with the no-copy bank under the map
  prefill -- candidates: the map's whole-layer wrap buffers (ds4_gpu_wrap_model_range over 3.6 GB per
  layer: are they freed between layers/frontiers?), the residency set publish after prefill
  (prefill pins, ds4_gpu_stream_expert_cache_prefill_begin with threshold 0 = always pinned), and the
  7.12 GiB prefill expert reserve. Reproduce with bench256.sh at BUDGET=30GB CTXMAX=98304 and watch
  `vm_stat` wired per layer with DS4_METAL_GRAPH_PREFILL_PROFILE=1; the fix is a prerequisite for any
  upstream PR from this branch beyond the three already open.
