#!/bin/zsh
# usage: eval_cfg.sh NAME MODEL [KEEPFILE]   -- runs the core suite subset, writes /tmp/eval/NAME.{log,trace}
NAME=$1; MODEL=$2; KEEP=$3
cd /Users/gaineyllc/ds4; mkdir -p /tmp/eval
export DS4_METAL_STREAM_EXPERT_NOCOPY=1
# DS4_SSD_CACHE_AUTO_PCT=100 (no decode-bank shrink) OOMs the GPU on the second session with a churning bank; keep-list runs set it explicitly.
[ -n "$KEEP" ] && export DS4_EXPERT_KEEP=$KEEP
Q=${EVAL_QUESTIONS:-24}; T=${EVAL_TOKENS:-3072}
./ds4-eval --plain --pause-ms 1 -m $MODEL --metal --ssd-streaming -c 16384 --suite core --questions $Q -n $T \
  --trace /tmp/eval/$NAME.trace > /tmp/eval/$NAME.log 2>&1
echo "EVAL_DONE $NAME"; tail -5 /tmp/eval/$NAME.log
