# DeepSeek V4.1 Flash on ds4 / M5 Max 128 GB — Handoff 2

**Continues:** `HANDOFF-v41-decode.md` (`f4852a0`). Read that first; this only records what changed.
**Branch:** `spec/engram-offload-engine`, `1a1f4a6` → `591fcd9`.
**Session goal as set:** get decode as close as possible to 47 tok/s, holding at 1M context, tested at 8k/16k, `ds41f-q2` only.

**Result: 14.15 → ~17.0 t/s, entirely from one knob. All three of handoff-1 §4's levers are now measured and closed. 47 is not reachable on this machine by engineering.**

---

## 1. The one thing that worked

Every A/B in handoff-1, and every run before this session, pinned
`--ssd-streaming-cache-experts 36GB` for comparability. Nobody had swept it.

| bank | median t/s | vs 36 GB | prefill | peak wired |
|---|---|---|---|---|
| 36 GB | 14.15 | — | ~157 | 55 G |
| 48 GB | 15.41 | +8.9% | ~150 | 57 G |
| **64 GB** | **16.98** | **+20.0%** | ~133 | 86 G |
| 68 GB | 16.84 | +19.0% | ~138 | 90 G |
| 72 GB | killed | — | — | 93 G |

Output byte-identical at every size. Costs ~13% of prefill. `watch2.sh` kills at
92 G wired, so 72 GB cannot run; 68 is within noise of 64 and leaves less
headroom. **Use 64 GB.** `~/ds4-bench/ab16k_bank.sh` honours `DS4_BANK`.

Do not push to 84/92 GB: the curve has already turned over, and that region is
the wedged-machine state handoff-1 §7 describes.

## 2. Why the curve turns over — and what the stall actually is

`DS4_METAL_V41_STREAM_STATS=1` prints the residency counters on the CLI path for
the first time (they existed, but only `ds4_gpu_print_memory_report` read them,
and the V4.1 CLI never calls it).

    36 GB bank   commits=372  total=2052.9 ms  avg=5.52 ms   7.3% of decode
    64 GB bank   commits=372  total=3435.7 ms  avg=9.24 ms  14.6% of decode

One `[MTLResidencySet commit]` per generated token, against a **628 µs median GPU
time per command buffer**. The count is already minimal — `residency_publish()`
has a dirty check, commits at most once per token, and deliberately omits
`requestResidency` (a comment records 18.2 → 8.6 t/s if you add it back).

Cost scales with the allocations being registered, so a bigger bank buys fewer
misses and pays a linearly growing tax. 64 GB is where they cross.

Command-buffer profile at 64 GB (`cbstat.py`, `cbtail.py` over
`DS4_METAL_CB_TIMES=1`), 1799 decode buffers:

    gpu    total 1214.5 ms   p50  628 us   p90 1097 us   max  6068 us
    gap    total 1662.4 ms   p50  167 us   p90 2973 us   max 18705 us
    driver total  815.8 ms   p50   27 us   p90 1714 us   max 10990 us

GPU idle between buffers exceeds GPU busy. The top 10% of buffers by gap carry
54.9% of all idle; their encode is 10× the rest (2362 vs 241 µs) and their idle
exceeds their own encode by 2710 µs. **The stall is tail-dominated — average-case
optimisation provably does not move it.**

**Ceiling: eliminating residency commits entirely returns 14.6%, ~17.0 → 19.9 t/s.**

## 3. Prediction is closed, and here is why

Added `DS4_V41_PREROUTE_DUMP` (one record per position/layer: actual top-6 plus
the post-FFN-norm hidden state, fp16) and analysed 15522 pairs.

| method | top-6 recall |
|---|---|
| Edge0 next-layer-router proxy | 23.1% |
| same-layer previous token (ds4 prefetches this today) | 25.3% |
| static per-layer top-6 by frequency | 27.3% |
| ridge head on the hidden state, held out | 28.6% |
| static per-layer top-12 | 40.3% |
| *bar for a hint to pay (handoff-1 §8)* | *~50%* |

A static popularity list with no features beats both the Edge0 warm start and
the baseline the engine already uses. A trained head barely beats the static
list.

Then the decisive measurement, before building the prefetcher:

    popularity prefetch headroom: top-12 per layer, 89760 checked
    over 7480 layer-samples, 12 not resident (0.0%)

**Twelve out of eighty-nine thousand.** Eviction is frequency-with-decay, so the
hottest experts per layer are exactly the ones the bank keeps. Popularity
predicts well *because* the predictable part of routing never stalls. The misses
are the tail, and no predictor fit to the head can name the tail.

## 4. Everything measured negative (do not retry)

All interleaved medians, 3 reps, at the bank noted.

| change | result |
|---|---|
| Engram zero-copy, 64 KiB windows | 14.45 → 10.47 (`1a1f4a6`) |
| Engram zero-copy, 16 KiB windows | 14.45 → 11.06 |
| early miss reads (`MTLSharedEvent` ring) | 13.89 → 13.72 (`c9a0a9c`) |
| `DEFER_BACKOFF_MAX=16` | 13.92 → 13.72, redos 5 → 12 |
| copy mode at 36 GB | 14.24 → 13.29 (`46e009f`) |
| `RESIDENCY_COMMIT_EVERY=3` | 15.90 → 15.67 (`1b42d1d`) |
| per-layer residency sets | not built — addresses the 15% fixed term, not the 85% |
| static top-12 prefetch | not built — 0.0% headroom |

## 5. Corrections to handoff-1

- **`requestResidency` on a file-backed no-copy buffer wires the pages.** Binding
  both 94.42 GiB Engram tables drove wired 4 → 93 G in 20 s and the watchdog
  killed the run before prefill. There is no "zero resident memory growth" once
  the GPU reads it. This is why the pasted spec's monolithic mapping is dead.
- **Under no-copy the copy-mode cache counters are never touched** (`entries=0`,
  zero lookups). The "55% bank residency" figure everything was reasoned from
  does not describe a no-copy run.
- **The deferred-expert path is dormant**: `deferred tokens=0 redos=5
  backed_off=181`, and a 64 GB bank does not wake it. Handoff-1 §5.1 items 1 and
  2 both target ~3% of batches. The 64 early-miss layer signals in a 400-token
  run are ~1.6 deferred passes out of 187.
- **Copy mode's +20–30% was a 30 GB result.** At 36 GB it is −6.7%, because
  copies halve effective residency — which handoff-1 §3 itself warned.
- **Cross-session batching already exists** for V4.1
  (`ds41_sessions_batch_supported`). What did not exist was §5.5's cap.
- **Timing drift is ~10–14% within a session**, monotonic with compressor fill
  (0 → 10.8 GiB over this session), and it moves in both directions. Sequential
  A/B is worthless. `abab.sh` interleaves and gets within-arm spread under 0.5%.
  **Reboot before measuring absolutes.**

## 6. Tooling added (`bench/`, also in `~/ds4-bench/`)

- `abab.sh` — interleaved A/B/A/B, medians. `ABAB_RUNNER` overrides the runner.
- `ab16k_bank.sh` (`DS4_BANK`), `ab16k_mode.sh` (no hardcoded `NOCOPY`),
  `banksweep.sh`, `banksweep2.sh`
- `cbstat.py`, `cbtail.py` — command-buffer time profile and tail attribution
- `astra_bytediff.py` — generated-text identity, the only correctness signal
- `preroute_collect.sh`, `preroute_analyze.py` — routing data (needs
  `/opt/homebrew/bin/python3`; system python3 has no numpy)
- `tools/engram_msl_check.m` — compiles `metal/*.metal` through the runtime
  compiler; the Metal toolchain is not installed on this machine
- env: `DS4_METAL_V41_STREAM_STATS`, `DS4_V41_BATCH_EXPERT_STATS`,
  `DS4_V41_PREROUTE_DUMP`, `DS4_ENGRAM_ZEROCOPY`,
  `DS4_METAL_V41_EARLY_MISS_READS`, `DS4_METAL_V41_RESIDENCY_COMMIT_EVERY`,
  `DS4_V41_BATCH_MAX_DISTINCT`

## 7. Open, and needing a decision

- **Dead code.** `ds4_gpu_engram_gather_gate()` and
  `kernel_engram_fused_gather_gate{,_store}` compile and have **zero callers** —
  ds4 applies the gate downstream in the graph, so wiring them would break
  parity. `gguf-tools/deepseek41_engram_serialize.py` is verified on synthetic
  data only and nothing consumes its format. Recommend deleting all four: they
  were spec-completion for a premise that turned out wrong.
- **`ds4_gpu_stream_expert_batch_row_limit`** is wired into
  `ds41_sessions_batch_supported` but only reachable with ≥2 concurrent
  sessions, so the single-stream bench never consults it. Needs `bp_run.sh`.
- **The distinct-expert measurement is redo-biased.** It only samples passes
  where the host reads the router back, which under deferral is redos only, so
  `missing/layer` is high. `distinct/layer` should carry over.

## 8. The calibration

Handoff-1 §8 put the honest ceiling at 20–25 t/s and said 47–50 needs the routed
experts to fit. That estimate survives contact:

- **fit more of the bank** — swept, 64 GB optimum, +20%. Exhausted.
- **make stops cheaper** — residency commits, 14.6% of decode, 85% irreducible.
  Ceiling ~19.9 t/s. Exhausted.
- **know routing earlier** — every predictor 23–29% against a ~50% bar, and the
  popular experts are already resident. Closed.

All three close on the same fact: **the predictable, cacheable part of this
workload is already handled, and what remains is the tail of the expert
distribution.** 142 GiB of routed experts against a 92 GB usable bank is the
whole problem.

47 t/s needs the experts to fit — a 192–256 GB machine — or a sub-2-bit format
that halves 142 GiB without wrecking the 71–74% acceptance rate. That is a
hardware or quantization decision, not an engineering one.
