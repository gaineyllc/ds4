#!/bin/zsh
# usage: srv_bench.sh LABEL MODEL KEEPFILE [N=4] -- warm-bank resident-regime decode t/s via ds4-server
LABEL=$1; MODEL=$2; KEEP=$3; N=${4:-4}
cd /Users/gaineyllc/ds4
export DS4_METAL_STREAM_EXPERT_NOCOPY=1 DS4_SSD_CACHE_AUTO_PCT=100 DS4_EXPERT_KEEP=$KEEP ${=EXTRA_ENV} DS4_METAL_GPU_BUSY_PROFILE=1 DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1
pkill -INT -x ds4-server 2>/dev/null; sleep 2
./ds4-server -m $MODEL --metal --ssd-streaming --ctx 16384 --host 127.0.0.1 --port 8111 > /tmp/srv_$LABEL.log 2>&1 &
SP=$!
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8111/v1/models >/dev/null 2>&1 && break; sleep 1; done
./bench/req.sh 8111 200 $N > /dev/null 2>&1
kill -INT $SP; wait $SP 2>/dev/null
echo "== $LABEL"
grep "gen=200 THINKING decoding chunk" /tmp/srv_$LABEL.log | awk '{for(i=1;i<=NF;i++) if($i ~ /^avg=/) print $i}' | tr '\n' ' '; echo
grep -E "gpu busy accum" /tmp/srv_$LABEL.log | tail -1
grep -E "deferred expert" /tmp/srv_$LABEL.log | tail -1 | cut -c1-120
