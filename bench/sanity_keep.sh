#!/bin/zsh
# keep-list sanity: K=384 must be byte-identical to no keep-list; K=376 (drop 8 least-used/layer) must stay coherent.
cd /Users/gaineyllc/ds4
export DS4_METAL_STREAM_EXPERT_NOCOPY=1 DS4_SSD_CACHE_AUTO_PCT=100
M=gguf/DeepSeek-V4.1-Flash-Q2.gguf
P="Find the sum of all integer bases b>9 for which 17_b divides 97_b. Think step by step."
./ds4 --ssd-streaming -m $M -p "$P" -n 200 --temp 0 2>&1 | grep -v "^ds4:" | grep -v "^processing" > /tmp/san_none.txt
DS4_EXPERT_KEEP=bench/keep/keep384.txt ./ds4 --ssd-streaming -m $M -p "$P" -n 200 --temp 0 2>&1 | grep -v "^ds4:" | grep -v "^processing" > /tmp/san_384.txt
DS4_EXPERT_KEEP=bench/keep/keep376.txt ./ds4 --ssd-streaming -m $M -p "$P" -n 200 --temp 0 2>&1 | grep -v "^ds4:" | grep -v "^processing" > /tmp/san_376.txt
md5 /tmp/san_none.txt /tmp/san_384.txt /tmp/san_376.txt
echo SANITY_DONE
