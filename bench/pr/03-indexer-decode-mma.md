# metal: a decode-shaped V4.1 indexer kernel (rows kernel + simdgroup-matrix version)

## What

Decode at long context ran the prefill-shaped indexer (`kernel_glm_indexer_scores_tiled`
/ `_batch`) for 1-3 tokens over hundreds of thousands of compressed rows: one threadgroup
per key tile, with the q side redone per tile. Two kernels for the decode shape:

- `kernel_dsv41_indexer_scores_decode_rows`: one dispatch per token, 8 simdgroups per
  threadgroup, each streaming a run of rows; q for the token lives in threadgroup memory.
- `kernel_dsv41_indexer_scores_decode` (default): C(8 rows x 32 heads) = K(8x128) .
  Q^T(128x32) from simdgroup_float8x8 tiles -- 16 key tiles read once from the cache and
  64 q tiles from threadgroup memory per 8 rows, then the per-row relu/weight reduction.
  ~25 instructions per row and lane against ~280 for the row-streaming kernel.

Taken for n_tokens < 8 at n_comp >= 4096 (the tiled kernel stays for prefill batches).
`DS4_METAL_V41_INDEX_DECODE_ROWS=1` selects the rows kernel; `DS4_METAL_DISABLE_V41_INDEX_DECODE=1`
the old path.

## Measured

M5 Max 128 GiB, macOS 26.5, quiet GPU, standalone harness (K f32, 32 heads x 128):

| rows x tokens | rows kernel      | MMA kernel       |
|---------------|------------------|------------------|
| 435000 x 3    | 0.72 ms / token  | 0.50 ms / token  |
| 869000 x 3    | 1.45 ms / token  | 1.09 ms / token  |

The MMA kernel reads 222 MB of keys per token at 435k in 0.50 ms (445 GB/s): it is at
the cache-read floor; the remaining cost of the indexer at 1M is that every token reads
K again. Worst |diff| against a double reference 8e-10 (both). On the 869k-context
DSpark decode timeline the indexer went from the largest single kernel to 23 ms of a
~160 ms GPU cycle (8 calls of ~2.9 ms for 2-3 tokens, GPU shared with a backup at the
time). Greedy 16k output byte-identical to the tiled path with DS4_METAL_V41_INDEX_SCALAR=1.
