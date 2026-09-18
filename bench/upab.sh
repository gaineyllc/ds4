#!/bin/zsh
# Interleaved upstream vs ours, identical config: plain decode (upstream cannot
# run --dspark on V4.1), copy mode, 64GB bank, same prompt.
reps=${1:-3}
: > ~/ds4-bench/upab.txt
for i in $(seq 1 $reps); do
  for who in up ours; do
    pkill -9 -x ds4 2>/dev/null; sleep 2
    if [[ $who == up ]]; then d=/Users/gaineyllc/ds4-main; else d=/Users/gaineyllc/ds4; fi
    cd $d; rm -f /tmp/ds4.lock
    (sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &
    ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming -c 16384 \
      --ssd-streaming-cache-experts 64GB --prompt-file ~/ds4-bench/p10k.txt \
      -n 400 --temp 0 > ~/ds4-bench/upab_${who}$i.txt 2>&1
    echo "$who$i $(grep -aoE 'prefill: [0-9.]+ t/s, generation: [0-9.]+' ~/ds4-bench/upab_${who}$i.txt | tail -1)" | tee -a ~/ds4-bench/upab.txt
  done
done
