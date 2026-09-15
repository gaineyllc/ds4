#!/bin/zsh
# Collect routing usage: prefill (layers 0-19 full, 20-39 tail) + 400 decode tokens per prompt.
cd /Users/gaineyllc/ds4
export DS4_METAL_STREAM_EXPERT_NOCOPY=1 DS4_SSD_CACHE_AUTO_PCT=100
mkdir -p /tmp/usage
for f in bench/calib/p*.txt; do
  b=$(basename $f .txt)
  [ -s /tmp/usage/$b.txt ] && continue
  DS4_EXPERT_USAGE_DUMP=/tmp/usage/$b.txt ./ds4 --ssd-streaming -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --prompt-file $f -n 400 --temp 0 -c 65536 > /tmp/usage/$b.log 2>&1
  grep -E "prefill:|generation:" /tmp/usage/$b.log | sed "s/^/$b /"
done
echo CALIB_DONE
