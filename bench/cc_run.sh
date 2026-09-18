#!/bin/zsh
# Concurrent-session DSpark check.
# usage: cc_run.sh <tag> <n_sessions> [ENV=VAL ...]
tag=$1; n=$2; shift 2
cd /Users/gaineyllc/ds4
pkill -9 -x ds4-server 2>/dev/null; pkill -9 -x ds4 2>/dev/null; sleep 2
log=~/ds4-bench/cc_$tag.log
env "$@" DS4_METAL_STREAM_EXPERT_NOCOPY=1 \
  ./ds4-server -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-native.gguf \
  --metal --ssd-streaming -c 16384 --ssd-streaming-cache-experts 64GB \
  --dspark --batched-session $n --port 8020 > $log 2>&1 &
for i in $(seq 1 240); do grep -qa "listening on" $log 2>/dev/null && break; sleep 1; done
grep -qa "listening on" $log || { echo "SERVER FAILED TO START"; tail -5 $log; exit 1; }
echo "--- startup ---"; grep -a -iE "DSpark|speculative|batched mode enabled" $log | head -4
t0=$(python3 -c 'import time;print(time.time())')
for r in A B; do
  [ "$r" = "B" ] && [ "$n" -lt 2 ] && continue
  curl -s --max-time 900 http://127.0.0.1:8020/v1/completions -H 'Content-Type: application/json' \
    -d @$HOME/ds4-bench/cc_req$r.json > ~/ds4-bench/cc_${tag}_$r.json &
done
wait
t1=$(python3 -c 'import time;print(time.time())')
pkill -9 -x ds4-server; sleep 2
echo "requests done"
import json,sys,os
