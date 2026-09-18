# HANDOFF v4.1 decode 3 — what the profiler changed

Continues HANDOFF-v41-decode-2.md. Everything here is from 2026-09-18 on the
M5 Max / 128 GB, `ds41f-q2`, 16k context, 64 GB bank, DSpark.

## Throughput

~16.4–16.7 t/s generation, ~139 t/s prefill. Unchanged from handoff-2's
headline: the 36→64 GB bank remains the only change that ever moved it.

## The correction that matters

Handoff-2 and commit 1f0d08a named residency-set commits as the decode stall,
14.6% of decode, with a ~19.9 t/s ceiling from removing them. **Wrong.** The
3.19 s of commits was divided by the decode duration while the commits happen
in prefill. Sampling profiler, main thread, same run shape
(`bench/profiles/{prefill,decode}.sample`):

|                                   | prefill | decode |
|-----------------------------------|---------|--------|
| `-[IOGPUMetalResidencySet commit]`| 31.10%  |  0.15% |
| `residency_publish`               | 15.55%  |  0.01% |
| `ds4_gpu_end_commands` (drain)    | 39.31%  | 58.48% |
| main-thread samples               |    7319 |  11161 |

Consequences: `RESIDENCY_COMMIT_EVERY` and the sharded residency set were both
aimed at a cost decode does not pay. Their numbers stand; their target does
not. And prefill is the larger half of wall clock at 16k (~41 s vs ~25 s) yet
`abab.sh` reads only `generation:` — every A/B this session was blind to it.

## Decode is drain-bound, and it routes back to capacity

    ds4_session_ds41_dspark_cycle              92.9%
     ds41_verify_suffix_tops                   89.1%
      ds41_graph_prefill_sweep                 80.3%
       ds41_moe_batch                          79.9%
        ds4_gpu_routed_moe_batch_tensor        58.5%   ds4_metal.m:48030
         ds4_gpu_end_commands -> waitUntilCompleted
    ds41_dspark_draft_pending                   5.9%
    stream_expert_cache_prepare_selected_batch  5.7%
    engram read_batch                           2.6%

58.5% of decode is the main thread blocked at the per-layer drain in the
**non-deferred** MoE path. That is what the deferred expert path exists to
remove. Deferral runs on **5 of 186 verify batches (2.7%)**: 44% of layers
miss an expert at a 64 GB bank, every miss costs a full-batch redo, and the
backoff escalates to cooldown=128 and keeps doubling.
`DS4_METAL_V41_DEFER_NO_BACKOFF=1` is −30% (15.28 → 10.61), so the backoff is
correct. Both halves of the run are bounded by the bank not fitting.

## Closed this session, with mechanisms

- **Residency sharding.** Commit cost is ≈ c·√(members), not linear, and churn
  dirties ~5–6 shards regardless of K. Flat across K=1/4/16/32
  (8.50/9.19/8.76/9.25 ms per publish). Dead at the driver anyway:
  `IOGPUMetalCommandQueue: command queue residency set limit of 32 exceeded`.
- **Commit batching.** `COMMIT_EVERY=8` cuts commits 372→217 and total time
  not at all (3411.7 → 3430.9 ms; per-commit 9.17 → 15.81 ms).
- **The SSD is not the decode bottleneck.** See `bench/ioprobe.c`. The device
  gives 4,980–10,634 MB/s *while decode runs*; decode's 2.0 GB/s is demand,
  not a ceiling. The no-copy bank issues no readahead and that is correct —
  the driver faults asynchronously, overlapped with GPU work. Four explicit
  warm variants all ≤ baseline; worst 9.06 t/s.
- **`mincore()` finding worth keeping:** 29,559 of 44,250 installed ranges
  (91.9 GiB) are already resident. Decode installs ~80 GiB of experts but only
  ~26 GB leaves the device — two thirds of installs re-wrap cached pages.
- **Pool width:** `pread_thread_count()` is `min(limit, n_tasks)` and a
  single-expert install builds exactly 3 tasks, so it never exceeds 3 of the
  pool's 9 threads. Chunking would fix it; nothing left wants the width.
- **Decode-time expert installs come from one site**, 8,627 per run (the batch
  selected-addr path), not `cache_install()`. Two earlier hook attempts missed
  decode entirely — instrument the call sites before hooking.

## Not a leak

82.4 GB wired during a run, 5.7 GB thirty seconds after exit, swap untouched
(27 MB). ds4 returns everything. The apparent "needs a reboot" degradation was
self-inflicted: an `ioprobe` run faulted across 365 GB of GGUF, left the
compressor at 18 GB, and halved throughput for ~90 minutes. It drained on its
own and the next clean run gave 16.70 t/s.

**Harness fix owed:** `abab.sh` must record compressor occupancy per sample and
report `prefill:` alongside `generation:`. A contaminated run is currently
invisible, and a prefill-only win is unmeasurable.

## Open

1. Prefill is 41 s of a 67 s run and 31% of it is one call that resists
   batching and sharding. The only untried lever is keeping cold experts out
   of the residency set and naming them per-dispatch with `useResource` — the
   code claims that cost "several milliseconds per layer", which predates the
   64 GB bank and is worth re-measuring.
2. Partial deferral: on a miss, roll back to the missing layer instead of to
   layer 0. Expected first miss is layer ~2.3, so this saves ~2–3 drains of 40
   per batch (~6%), and needs real surgery on `ds41_verify_rollback`.
3. 47 t/s came from a configuration where the deferred path was live
   (`ds4_engram.c`: "with the deferred expert path ... the pool is worth +12%,
   42.8 -> 47.9 t/s"). Getting back there means making the bank fit.
