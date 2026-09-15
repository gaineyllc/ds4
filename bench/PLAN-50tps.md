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
