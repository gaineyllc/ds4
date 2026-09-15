# server: find token-text disk stores by the request's rendered tokens

## What

A token-text KV disk store (`key=token-text`) is keyed by the *rendered* text of the
session's tokens, and rendering is not the prompt text: a literal `<|system|>` inside a
source file tokenizes to the special token and renders back as `<｜System｜>`. So the cold
store of a 570k-token prompt never matched the same prompt again, and every server
restart re-prefilled the whole prompt.

When the text lookup misses, look the request's own tokens up rendered the same way; on
a hit, continue with the request's tokens (the stored prefix must be exactly theirs)
instead of re-tokenizing rendered text.

`tests/tok_roundtrip.c` (make tests/tok_roundtrip) checks token-prefix rendering against
a prompt file.

## Measured

M5 Max 128 GiB, macOS 26.5, DeepSeek V4.1 Flash Q2, `--ssd-streaming -c 1048576
--kv-disk-dir ... --kv-cache-cold-max-tokens 2000000`. The second 869k-token prefill of
the day resumed from the 570401-token cold store and prefilled 570401 -> 868904 in 13.7
min, where before the change every restart re-prefilled the whole prompt (25-38 min).
`./ds4_test --server`: OK on upstream main + this change (branch pr/server-disk-store-rendered-lookup). No kernel or backend code touched.
