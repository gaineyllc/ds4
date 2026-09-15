#!/bin/zsh
cd /Users/gaineyllc/ds4
export EVAL_QUESTIONS=${EVAL_QUESTIONS:-12} EVAL_TOKENS=${EVAL_TOKENS:-2048}
./bench/eval_cfg.sh denseq4k_keep176 gguf/DeepSeek-V4.1-Flash-Q2-denseQ4K.gguf bench/keep/keep176.txt
./bench/eval_cfg.sh q2_keep176      gguf/DeepSeek-V4.1-Flash-Q2.gguf          bench/keep/keep176.txt
./bench/eval_cfg.sh q2_full         gguf/DeepSeek-V4.1-Flash-Q2.gguf
# keep160 dropped: keep176 already loops
echo EVAL_ALL_DONE
