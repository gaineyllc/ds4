#!/bin/zsh
# Interleaved copy vs no-copy at a fixed bank, scored on BOTH prefill and
# generation -- abab.sh reads only generation:, which is exactly the number
# that cannot see this tradeoff.
reps=${1:-3}; bank=${2:-64GB}
: > ~/ds4-bench/modeab.txt
for i in $(seq 1 $reps); do
  for m in copy nocopy; do
    pkill -9 -x ds4 2>/dev/null; sleep 2
    cd /Users/gaineyllc/ds4; rm -f /tmp/ds4.lock
    (sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &
    if [[ $m == nocopy ]]; then ev=(DS4_METAL_STREAM_EXPERT_NOCOPY=1); else ev=(); fi
    env "${ev[@]}" ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
      --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-native.gguf \
      --metal --ssd-streaming -c 16384 --ssd-streaming-cache-experts $bank --dspark \
      --prompt-file ~/ds4-bench/p10k.txt -n 400 --temp 0 \
      > ~/ds4-bench/mode_${m}$i.txt 2>&1
    echo "$m$i $(grep -aoE 'prefill: [0-9.]+ t/s, generation: [0-9.]+' ~/ds4-bench/mode_${m}$i.txt | tail -1)" | tee -a ~/ds4-bench/modeab.txt
  done
done
