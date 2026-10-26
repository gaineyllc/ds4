#!/bin/zsh
# Start ds4-server with the production settings this fork was tuned with
# (M5 Max 128 GB, Oct 2026). Run from anywhere; it cds into the repo.
# Override any variable in the environment or in ~/.ds4/mac.env.
set -u
REPO=${DS4_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
[ -f "$HOME/.ds4/mac.env" ] && source "$HOME/.ds4/mac.env"
: ${DS4_PORT:=8031}
: ${DS4_CTX:=1048576}            # 1M context, the model's native window
: ${DS4_BANK:=56GB}              # resident expert bank (128 GB Mac)
: ${DS4_SLOTS:=2}                # resident sessions; extra requests queue
: ${DS4_KVDIR:=$HOME/.ds4/kv}    # on-disk prompt cache
: ${DS4_KVMB:=40000}             # prompt-cache disk budget, MB
: ${DS4_LOG:=$HOME/.ds4/ds4-server.log}
# Expert bank: copied by default. Measured 2026-10-26 on an M5 Max (128 GB), production config
# (2 sessions, 1M ctx, DSpark, vision, 56 GB bank): warm decode 16.7 t/s copied vs 15.0 no-copy
# (ds4-bench, 1 session: 18.0 vs 14.2 at 54k-131k). No-copy is only faster on a cold bank.
# DS4_NOCOPY=1 makes the bank point at the model's own pages instead.
[ "${DS4_NOCOPY:-0}" = 1 ] && export DS4_METAL_STREAM_EXPERT_NOCOPY=1
export DS4_SSD_CACHE_AUTO_PCT=${DS4_SSD_CACHE_AUTO_PCT:-80}
export DS4_ENGRAM_TOKEN_READERS=${DS4_ENGRAM_TOKEN_READERS:-8}
export DS4_DECODE_SLICE_TOKENS=${DS4_DECODE_SLICE_TOKENS:-64}   # time-sliced concurrent sessions
M=gguf/DeepSeek-V4.1-Flash-1017-UNC-Q2.gguf
D=gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf
V=gguf/DeepSeek-V4.1-Flash-1017-Vision.gguf
cd "$REPO" || exit 1
for f in ds4-server "$M" "$D" "$V"; do
  [ -e "$f" ] || { echo "missing $REPO/$f; see docs/mac-setup/README.md" >&2; sleep 60; exit 1; }
done
mkdir -p "$DS4_KVDIR" "$(dirname "$DS4_LOG")"
exec ./ds4-server -m "$M" --metal --ssd-streaming -c "$DS4_CTX" \
  --ssd-streaming-cache-experts "$DS4_BANK" \
  --mtp-model "$D" --dspark --vision "$V" \
  --batched-session "$DS4_SLOTS" --mixed-prefill-quantum 8192 \
  --kv-disk-dir "$DS4_KVDIR" --kv-disk-space-mb "$DS4_KVMB" \
  --kv-cache-cold-max-tokens 2000000 --kv-cache-continued-interval-tokens 32768 \
  --host 127.0.0.1 --port "$DS4_PORT" >> "$DS4_LOG" 2>&1
