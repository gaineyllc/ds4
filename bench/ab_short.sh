#!/bin/zsh
cd /Users/gaineyllc/ds4
export DS4_METAL_STREAM_EXPERT_NOCOPY=1
M=gguf/DeepSeek-V4.1-Flash-Q2.gguf
echo "servers: $(pgrep -x ds4-server | wc -l | tr -d ' ')"
P="Explain in detail how a write-ahead log guarantees durability, and why fsync placement matters."
./ds4 --ssd-streaming -m $M -p "$P" -n 150 --temp 0 2>&1 | grep -v "^ds4:" > /tmp/a.txt
DS4_METAL_V41_DISABLE_DEFER_EXPERT_SYNC=1 ./ds4 --ssd-streaming -m $M -p "$P" -n 150 --temp 0 2>&1 | grep -v "^ds4:" > /tmp/b.txt
wc -c /tmp/a.txt /tmp/b.txt | head -2
diff /tmp/a.txt /tmp/b.txt > /dev/null && echo "GATE: IDENTICAL" || echo "GATE: DIFFER"
P="Write one paragraph explaining how a B-tree differs from a hash index."
for i in 1 2 3; do
  ./ds4 --ssd-streaming -m $M -p "$P" -n 60 --temp 0 2>&1 | grep -oE "generation: [0-9.]+|already running" | sed "s/^/defer $i /"
  DS4_METAL_V41_DISABLE_DEFER_EXPERT_SYNC=1 ./ds4 --ssd-streaming -m $M -p "$P" -n 60 --temp 0 2>&1 | grep -oE "generation: [0-9.]+|already running" | sed "s/^/sync  $i /"
done
echo "servers: $(pgrep -x ds4-server | wc -l | tr -d ' ')"
echo DONE
