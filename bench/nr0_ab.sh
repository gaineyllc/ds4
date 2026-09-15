#!/bin/zsh
# usage: nr0_ab.sh IQ2 Q2K Q4KD Q4KLOW  -- rebuild with these row counts and bench
cd /Users/gaineyllc/ds4
sed -i '' -e "s/^#define DS4_METAL_N_R0_IQ2_XXS_ADDR .*/#define DS4_METAL_N_R0_IQ2_XXS_ADDR $1/" \
  -e "s/^#define DS4_METAL_N_R0_Q2_K_SUM6 .*/#define DS4_METAL_N_R0_Q2_K_SUM6    $2/" \
  -e "s/^#define DS4_METAL_N_R0_Q4_K_DENSE .*/#define DS4_METAL_N_R0_Q4_K_DENSE   $3/" \
  -e "s/^#define DS4_METAL_N_R0_Q4_K_ATTN_LOW .*/#define DS4_METAL_N_R0_Q4_K_ATTN_LOW $4/" ds4_metal.m
make -j12 ds4-server 2>&1 | grep -iE "warning|error"
./bench/srv_bench.sh nr0_$1$2$3$4 gguf/DeepSeek-V4.1-Flash-Q2-denseQ4K.gguf bench/keep/keep176.txt 2 2>&1 | grep -E "avg=|busy" | tr '\n' ' '; echo " <- iq2=$1 q2k=$2 q4kd=$3 q4klow=$4"
