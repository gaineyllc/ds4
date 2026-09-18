#!/bin/zsh
# ab16k.sh without the hardcoded DS4_METAL_STREAM_EXPERT_NOCOPY=1, so copy mode
# can be measured. Pass NOCOPY explicitly to get the old behaviour.
# usage: ab16k_mode.sh <tag> [ENV=VAL ...]
tag=$1; shift
cd /Users/gaineyllc/ds4; rm -f /tmp/ds4.lock
(sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &
env "$@" ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-native.gguf --metal --ssd-streaming -c 16384 --ssd-streaming-cache-experts 36GB --dspark --prompt-file ~/ds4-bench/p10k.txt -n 400 --temp 0 > ~/ds4-bench/run16k_$tag.txt 2>&1
echo "$tag: $(grep -E 'prefill:' ~/ds4-bench/run16k_$tag.txt)"
