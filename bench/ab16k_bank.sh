#!/bin/zsh
# ab16k with the expert bank size taken from DS4_BANK (default 36GB), so the
# single biggest lever in handoff 4 -- "fit more of the bank" -- can be swept.
# usage: DS4_BANK=64GB ab16k_bank.sh <tag> [ENV=VAL ...]
tag=$1; shift
bank=${DS4_BANK:-36GB}
cd /Users/gaineyllc/ds4; rm -f /tmp/ds4.lock
(sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &
env "$@" DS4_METAL_STREAM_EXPERT_NOCOPY=1 ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-native.gguf --metal --ssd-streaming -c 16384 --ssd-streaming-cache-experts $bank --dspark --prompt-file ~/ds4-bench/p10k.txt -n 400 --temp 0 > ~/ds4-bench/run16k_$tag.txt 2>&1
echo "$tag ($bank): $(grep -E 'prefill:' ~/ds4-bench/run16k_$tag.txt)"
