#!/bin/zsh
# Interleaved A/B/A/B benchmark.
#
# Single-run comparisons on this box are worthless: identical binaries measured
# 14.36-16.44 t/s on 2026-09-17, a 14% spread, and it drifts monotonically
# within a session as the compressor fills. Interleaving makes run order a
# shared-mode error instead of the signal.
#
#   abab.sh <tag> <reps> -- <A env...> -- <B env...>
#   abab.sh ring 3 -- DS4_X=0 -- DS4_X=1
#
# Prints every sample, then each arm's median and the median delta. Median,
# not mean: one watchdog kill or one thermal dip should not decide the answer.

set -u
tag=$1; reps=$2; shift 2
[[ "${1:-}" == "--" ]] && shift
A=(); B=(); cur=A
for a in "$@"; do
  if [[ "$a" == "--" ]]; then cur=B; continue; fi
  if [[ $cur == A ]]; then A+=("$a"); else B+=("$a"); fi
done

echo "A: ${A[*]:-<none>}"
echo "B: ${B[*]:-<none>}"
out=~/ds4-bench/abab_$tag.txt
: > $out

runner=${ABAB_RUNNER:-~/ds4-bench/ab16k.sh}

gen() {  # $1 = run tag -> prints generation t/s
  grep -oE 'generation: [0-9.]+' ~/ds4-bench/run16k_$1.txt 2>/dev/null | tail -1 | awk '{print $2}'
}

for i in $(seq 1 $reps); do
  for arm in A B; do
    pkill -9 -x ds4 2>/dev/null
    sleep 2
    rt="${tag}_${arm}${i}"
    if [[ $arm == A ]]; then
      $runner $rt "${A[@]}" > /dev/null 2>&1
    else
      $runner $rt "${B[@]}" > /dev/null 2>&1
    fi
    g=$(gen $rt)
    echo "$arm$i ${g:-FAILED}" | tee -a $out
  done
done

echo "---" | tee -a $out
python3 - "$out" <<'PY' | tee -a $out
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip() and l.split()[0][0] in 'AB']
def med(arm):
    v = sorted(float(r[1]) for r in rows if r[0][0] == arm and r[1] != 'FAILED')
    if not v: return None
    n = len(v)
    return v[n//2] if n % 2 else (v[n//2-1] + v[n//2]) / 2
a, b = med('A'), med('B')
print('A samples', [r[1] for r in rows if r[0][0] == 'A'])
print('B samples', [r[1] for r in rows if r[0][0] == 'B'])
if a and b:
    print('A median %.2f  B median %.2f  delta %+.2f t/s (%+.1f%%)'
          % (a, b, b - a, 100.0 * (b - a) / a))
PY
