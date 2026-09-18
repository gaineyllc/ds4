#!/bin/zsh
# Collect a prerouter training set. No --dspark on purpose: the probe and the
# dump hang off ds41_graph_step_once, and with DSpark decode runs the verify
# batch instead, so a DSpark run yields one token's worth of records. Routing
# does not depend on DSpark, so the single-token path is the right collector.
# usage: preroute_collect.sh <tag> <n_tokens> [bank]
set -u
tag=$1; n=${2:-400}; bank=${3:-64GB}
cd /Users/gaineyllc/ds4; rm -f /tmp/ds4.lock
(sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &
env DS4_METAL_STREAM_EXPERT_NOCOPY=1 \
    DS4_V41_PREROUTE_PROBE=1 \
    DS4_V41_PREROUTE_DUMP=$HOME/ds4-bench/preroute_$tag.bin \
    ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming \
    -c 16384 --ssd-streaming-cache-experts $bank \
    --prompt-file ~/ds4-bench/p10k.txt -n $n --temp 0 \
    > ~/ds4-bench/run16k_$tag.txt 2>&1
echo "$tag: $(grep -E 'prefill:|prerouter dump:' ~/ds4-bench/run16k_$tag.txt | tr '\n' ' ')"
ls -lh ~/ds4-bench/preroute_$tag.bin 2>/dev/null | awk '{print "  file:", $5}'
