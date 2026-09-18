#!/bin/zsh
# Fine bank sweep between the last size that ran and the one the watchdog
# killed. usage: banksweep2.sh <reps> <bank> [bank...]
set -u
reps=$1; shift
banks=("$@")
out=~/ds4-bench/banksweep2.txt
: > $out
for i in $(seq 1 $reps); do
  for b in $banks; do
    pkill -9 -x ds4 2>/dev/null
    sleep 3
    t="bank2_${b}_$i"
    DS4_BANK=$b ~/ds4-bench/ab16k_bank.sh $t > /dev/null 2>&1
    g=$(grep -oE 'generation: [0-9.]+' ~/ds4-bench/run16k_$t.txt 2>/dev/null | tail -1 | awk '{print $2}')
    p=$(grep -oE 'prefill: [0-9.]+' ~/ds4-bench/run16k_$t.txt 2>/dev/null | tail -1 | awk '{print $2}')
    hr=$(grep -oE 'hit_rate=[0-9.]+' ~/ds4-bench/run16k_$t.txt 2>/dev/null | tail -1)
    pk=$(grep -oE 'wired=[0-9]+G' ~/ds4-bench/watch.log 2>/dev/null | tail -40 | sort -t= -k2 -n | tail -1)
    echo "$b rep$i gen=${g:-FAILED} prefill=${p:-} $hr peak_$pk" | tee -a $out
  done
done
echo "---" | tee -a $out
python3 - "$out" <<'PY' | tee -a $out
import sys, collections, re
by = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r'(\d+GB) rep\d+ gen=([0-9.]+)', l)
    if m: by[m.group(1)].append(float(m.group(2)))
for b in sorted(by, key=lambda x: int(x[:-2])):
    v = sorted(by[b]); n = len(v)
    med = v[n//2] if n % 2 else (v[n//2-1]+v[n//2])/2
    print('%-6s median %.2f t/s  samples %s' % (b, med, v))
PY
