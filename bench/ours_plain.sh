#!/bin/zsh
# Our branch, plain decode (no DSpark), same flags as the upstream baseline.
tag=$1; shift
cd /Users/gaineyllc/ds4; rm -f /tmp/ds4.lock
pkill -9 -x ds4 2>/dev/null; sleep 2
(sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &
env "$@" ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming -c 16384 --ssd-streaming-cache-experts 64GB --prompt-file ~/ds4-bench/p10k.txt -n 400 --temp 0 > ~/ds4-bench/ours_$tag.txt 2>&1
echo "$tag: $(grep -E 'prefill:' ~/ds4-bench/ours_$tag.txt)"
