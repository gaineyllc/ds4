#!/bin/zsh
# Upstream (origin/main) baseline, same prompt/model/context as ab16k_bank.sh.
# Upstream refuses --dspark on V4.1 ("DSpark ... not supported"), so the
# baseline is plain decode.
# usage: up_run.sh <tag> [extra ds4 args...] -- [ENV=VAL ...]
tag=$1; shift
args=(); envs=(); cur=args
for a in "$@"; do
  if [[ "$a" == "--" ]]; then cur=envs; continue; fi
  if [[ $cur == args ]]; then args+=("$a"); else envs+=("$a"); fi
done
cd /Users/gaineyllc/ds4-main; rm -f /tmp/ds4.lock
pkill -9 -x ds4 2>/dev/null; sleep 2
(sleep 5; ~/ds4-bench/watch2.sh) >/dev/null 2>&1 &
env "${envs[@]}" ./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming -c 16384 "${args[@]}" --prompt-file ~/ds4-bench/p10k.txt -n 400 --temp 0 > ~/ds4-bench/up_$tag.txt 2>&1
echo "$tag: $(grep -E 'prefill:' ~/ds4-bench/up_$tag.txt)"
