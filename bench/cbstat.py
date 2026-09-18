"""Summarize DS4_METAL_CB_TIMES lines: where a decode cycle's wall time goes.

Line shape:
  ds4: cb <label>: driver N us, queue-wait N us, gpu N us, gap-from-prev-gpu-end N us
"""
import re, sys, statistics as st

pat = re.compile(
    r'ds4: cb (.+?): driver ([0-9.]+) us, queue-wait ([0-9.]+) us, '
    r'gpu ([0-9.]+) us, gap-from-prev-gpu-end ([0-9.]+) us')

drv, qw, gpu, gap = [], [], [], []
for line in open(sys.argv[1], errors='replace'):
    m = pat.search(line)
    if not m:
        continue
    drv.append(float(m.group(2)))
    qw.append(float(m.group(3)))
    gpu.append(float(m.group(4)))
    gap.append(float(m.group(5)))

n = len(gpu)
if not n:
    print('no cb lines')
    sys.exit(0)

# skip prefill: take the last 60% as decode-dominated
cut = int(n * 0.4)
drv, qw, gpu, gap = drv[cut:], qw[cut:], gpu[cut:], gap[cut:]

def s(name, v):
    v2 = sorted(v)
    print('%-10s n=%d  total=%8.1f ms  mean=%7.1f us  p50=%7.1f  p90=%7.1f  max=%8.1f'
          % (name, len(v), sum(v) / 1000.0, st.mean(v),
             v2[len(v2)//2], v2[int(len(v2)*0.9)], v2[-1]))

print('command buffers analysed (last 60%%): %d' % len(gpu))
s('driver', drv)
s('queue-wait', qw)
s('gpu', gpu)
s('gap', gap)
tot = sum(drv) + sum(qw) + sum(gpu) + sum(gap)
print()
print('share of accounted wall time:')
for name, v in (('driver', drv), ('queue-wait', qw), ('gpu', gpu), ('gap', gap)):
    print('  %-10s %5.1f%%' % (name, 100.0 * sum(v) / tot))
print()
print('gap is time the GPU was idle between command buffers -- the stall.')
