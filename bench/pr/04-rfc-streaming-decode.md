# RFC: V4.1 Flash decode under SSD streaming on 128 GB Macs — no-copy bank, deferred expert sync, DSpark

Not a PR yet: this is the data and the shape of the changes, to agree on how to slice them.

## What the branch does (ssd-decode-bank-shrink, 57 commits on main 9139e2a)

1. **No-copy expert bank.** Streaming experts stay in the model's own mmap'd pages: a cache entry is three
   `newBufferWithBytesNoCopy` views (gate/up/down) over the file mapping, kept resident through one
   MTLResidencySet on the queue, instead of a pread into a wired slab. Installs copy nothing; the bank is a
   cap on entry count computed from RAM minus the context's buffers, the support model and a headroom
   (DS4_SSD_CACHE_HEADROOM_GIB, 24). Prefill rows are handed back to the OS after prefill (MADV_FREE_REUSABLE)
   and the bank grows into them for decode; the session rewinds instead of rebuilding (rewind ring of the last
   8192 raw rows).
2. **Deferred expert sync.** A decode token (and now a verify batch of up to 32 rows) runs all routed layers
   without stopping to read the router: the address kernels (kernel_mul_mv_addr_*) read the router's output
   and per-layer GPU address tables that install/evict keep current, raise a miss flag if an expert is absent,
   and the host rolls the token back and reruns it with the stops only then. Warm streak / backoff gate it.
3. **V4.1 Flash DSpark.** The drafter module converted to a support GGUF, batched verifier with rollback
   and partial commit, Markov head on the GPU, proposal from the committed row, key window seeding.
4. Smaller, separable: radix-select top-k (PR), decode-shaped indexer kernels (PR), disk-store lookup (PR),
   raw-log rewind, prefill rows release, tokenizer round-trip test.

## Numbers (M5 Max 128 GiB, macOS 26.5, V4.1 Flash Q2, Cursor/Codex/Chrome resident ~40 GB)

- ds4-bench, 2k→64k in 2k steps, 128 greedy tokens per frontier, same 23 GiB bank, branch vs main:
  generation 14.7 t/s mean vs 11.4 (+29%, every frontier); incremental 2k prefill 76.6 vs 111.3 (−31%),
  first token 330 ms vs 150. The prefill loss is the bank pinned through the sweep; unpinning it
  (DS4_METAL_STREAM_EXPERT_PIN_PREFILL_TOKENS) restores 128–131 t/s but decode pages the bank back in
  (3.7 s, then ~8 t/s for the next ~128 tokens). Trade-off, not a bug; default is pinned.
- 16k DSpark: 10.7–10.8 t/s (block 5, ~1.7 tokens per cycle), byte-identical with and without deferral.
- 869k context (server, 1M window, 47 GiB bank): 10.2 t/s first request after a 298k-token continued
  prefill at 531 t/s, 8–8.7 on rewind-based repeats. The bank holds ~50% of the experts per layer there,
  so a verify batch misses almost surely and the deferred path does not engage; the timeline is 161 ms of
  GPU per cycle (indexer 22, expert matvecs 33, dense ~40, tiny dispatches ~15) plus ~65 ms host.
- Memory: wired grows 1:1 with the bank; the machine hard-locked once (14:33) six minutes after a 16k run
  had held wired at 87 GiB with the compressor at 21 GiB — the bank sizing is the risky part of all this.

## Questions for upstream

- Is a Metal-only no-copy bank acceptable, or should it be behind a flag until CUDA has an equivalent?
- The deferred sync changes the failure mode from "stop and load" to "run, detect, redo": fine for decode
  (nothing is committed before the check), but it needs the residency set and the rewind ring. Slice: (a)
  the address tables + miss flag, (b) one-row deferral, (c) batches.
- DSpark for V4.1 is its own series (conversion script, verifier, head). It is the only path to >30 t/s at
  long context on this hardware, since a cycle's dense reads are shared by the rows it verifies.
