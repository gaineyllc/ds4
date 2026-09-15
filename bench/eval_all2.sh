#!/bin/zsh
cd /Users/gaineyllc/ds4
export EVAL_QUESTIONS=${EVAL_QUESTIONS:-12} EVAL_TOKENS=${EVAL_TOKENS:-2048}
while pgrep -f eval_all.sh >/dev/null; do sleep 20; done
./bench/sanity_keep.sh > /tmp/sanity.log 2>&1
DS4_SSD_CACHE_PERCENT=92 ./bench/eval_cfg.sh denseq4k_keep224 gguf/DeepSeek-V4.1-Flash-Q2-denseQ4K.gguf bench/keep/keep224.txt
./bench/eval_cfg.sh denseq4k_full gguf/DeepSeek-V4.1-Flash-Q2-denseQ4K.gguf
echo EVAL_ALL2_DONE
