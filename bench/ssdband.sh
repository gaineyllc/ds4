#!/bin/zsh
# Is decode SSD-bandwidth-bound?
#
# The host-side theories (40 drains per cycle, residency commits) only matter
# if the SSD is not already saturated. At 44% of layers missing an expert and
# 9.49 MiB per expert, the bank could be reading at device speed, in which case
# no amount of host scheduling changes anything and the answer is bytes/token.
#
# One iostat stream at 1 Hz, epoch-stamped, so decode can be told from prefill.
# usage: DS4_BANK=64GB ssdband.sh <tag>
tag=${1:-band}
bank=${DS4_BANK:-64GB}
cd /Users/gaineyllc/ds4; rm -f /tmp/ds4.lock
out=~/ds4-bench/run16k_$tag.txt
io=~/ds4-bench/io_$tag.txt

iostat -d -w 1 disk0 | while IFS= read -r l; do print -r -- "$(date +%s) $l"; done > $io 2>&1 &
iopid=$!
(sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &

echo "MARK start $(date +%s)" >> $io
env DS4_METAL_STREAM_EXPERT_NOCOPY=1 DS4_METAL_V41_STREAM_STATS=1 \
  ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-native.gguf \
  --metal --ssd-streaming -c 16384 --ssd-streaming-cache-experts $bank \
  --dspark --prompt-file ~/ds4-bench/p10k.txt -n 400 --temp 0 > $out 2>&1
echo "MARK end $(date +%s)" >> $io

sleep 1; kill $iopid 2>/dev/null
grep -E "prefill:" $out
