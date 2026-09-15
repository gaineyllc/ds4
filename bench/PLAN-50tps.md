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
