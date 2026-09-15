# metal: radix-select top-k for the V4.1 indexer

## What

The indexer's top-k (`kernel_argsort_f32_i32_desc` + merge) keeps every 1024-row block's
top 512 and merges them in log2(blocks) dependent passes: at 435k rows that is 9 passes
over 217k survivors, 2-3 ms per call, and the 32-row causal batches of prefill took
9.6 ms.

`kernel_topk_select_{init,hist,scan,compact,finish}` (metal/argsort.metal) find the k-th
key exactly with four 8-bit histogram levels over the sortable-int form of the scores,
compact the winners in one pass, and let one threadgroup order them (score desc, index
asc), so the selected set and its order are the ones the sort produced. Taken when
n_comp >= 4096, top_k <= 2048 and every token has at least top_k visible rows; the sort
stays for everything else. `DS4_METAL_DISABLE_TOPK_SELECT=1` restores the sort.

Only when more than 2048 rows tie exactly at the k-th boundary is the choice among the
tied rows arrival-ordered instead of index-ordered (tests/test_metal_topk_select.m covers
random rows and rows full of ties against the sort).

## Measured

M5 Max 128 GiB, macOS 26.5, DeepSeek V4.1 Flash Q2, Metal SSD streaming.
- Standalone, quiet GPU (tests/test_metal_topk_select, best of the loops; sort = DS4_METAL_DISABLE_TOPK_SELECT=1):

  | rows x tokens, k            | sort     | select   |
  |-----------------------------|----------|----------|
  | 65537 x 1, k=512            | 0.89 ms  | 0.37 ms  |
  | 435000 x 1, k=512           | 2.36 ms  | 0.40 ms  |
  | 435000 x 3, k=512           | 2.27 ms  | 0.45 ms  |
  | 869000 x 1, k=512           | 3.34 ms  | 0.41 ms  |
  | 54000 x 1, k=2048           | 1.12 ms  | 0.53 ms  |
  | 435000 x 32 causal (prefill)| 4.94 ms  | 1.82 ms  |
  | 435000 x 1, all ties        | 0.82 ms  | 0.60 ms  |
  | 4096 x 1, k=512             | 0.83 ms  | 0.71 ms  |
- Encoder timeline of a 869k-context decode: kernel_topk_select_hist 58 us x 7348 calls =
  428 ms over the prefill tail where the causal argsort was 3.4 ms per batch.
- Decode: 435k rows / 3 tokens 3.0 -> 0.35 ms per call, 869k 2.7 -> 0.43 ms.
- Greedy output at 16k with DSpark byte-identical with and without the select.
- `./ds4_test --metal-kernels`: OK.
