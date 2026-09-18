#!/bin/zsh
# Sweep the expert bank size, interleaved so session drift cannot order the
# result. Handoff 4: "fit more of the bank" is one of only three levers, and
# it is the one nobody has swept -- every A/B so far pinned 36GB for
# comparability. 128 GB machine, ~52.7 GiB planned at 36GB bank, so there is
# real headroom. watch2.sh still kills at 92G wired.
set -u
reps=${1:-2}
banks=(36GB 48GB 60GB 72GB)
out=~/ds4-bench/banksweep.txt
: > $out
for i in $(seq 1 $reps); do
  for b in $banks; do
    pkill -9 -x ds4 2>/dev/null
    sleep 2
    t="bank_${b}_$i"
    DS4_BANK=$b ~/ds4-bench/ab16k_bank.sh $t > /dev/null 2>&1
    g=$(grep -oE 'generation: [0-9.]+' ~/ds4-bench/run16k_$t.txt 2>/dev/null | tail -1 | awk '{print $2}')
    h=$(grep -oE 'hit_rate=[0-9.]+' ~/ds4-bench/run16k_$t.txt 2>/dev/null | tail -1)
    w=$(grep -oE 'wired=[0-9]+G' ~/ds4-bench/watch.log 2>/dev/null | tail -1)
    echo "$b rep$i ${g:-FAILED} ${h:-} ${w:-}" | tee -a $out
  done
done
echo "---" | tee -a $out
python3 - "$out" <<'PY' | tee -a $out
import sys, collections
rows = [l.split() for l in open(sys.argv[1]) if l.strip() and l.split()[0].endswith('GB')]
by = collections.defaultdict(list)
for r in rows:
    if r[2] != 'FAILED':
        by[r[0]].append(float(r[2]))
for b in sorted(by, key=lambda x: int(x[:-2])):
    v = sorted(by[b]); n = len(v)
    med = v[n//2] if n % 2 else (v[n//2-1]+v[n//2])/2
    print('%-6s median %.2f t/s  samples %s' % (b, med, v))
PY
