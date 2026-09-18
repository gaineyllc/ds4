"""Attribute the decode stall tail.

gap[i] is GPU idle before command buffer i. driver[i] is the CPU-side encode
time for i. If the idle is the CPU building the next buffer, gap should track
driver. If gap greatly exceeds driver, the CPU was blocked on something else --
disk reads or a residency commit -- and that is a different fix.
"""
import re, sys, statistics as st

pat = re.compile(
    r'ds4: cb (.+?): driver ([0-9.]+) us, queue-wait ([0-9.]+) us, '
    r'gpu ([0-9.]+) us, gap-from-prev-gpu-end ([0-9.]+) us')

rows = []
for line in open(sys.argv[1], errors='replace'):
    m = pat.search(line)
    if m:
        rows.append((m.group(1), float(m.group(2)), float(m.group(3)),
                     float(m.group(4)), float(m.group(5))))

cut = int(len(rows) * 0.4)
rows = rows[cut:]
print('decode command buffers: %d' % len(rows))

gaps = sorted(rows, key=lambda r: -r[4])
thresh = gaps[int(len(gaps) * 0.10)][4]
tail = [r for r in rows if r[4] >= thresh]
rest = [r for r in rows if r[4] < thresh]

tail_gap = sum(r[4] for r in tail)
all_gap = sum(r[4] for r in rows)
print('top 10%% of buffers by gap carry %.1f%% of all idle (%.0f of %.0f ms)'
      % (100.0 * tail_gap / all_gap, tail_gap / 1000, all_gap / 1000))
print()
print('%-22s %10s %10s %10s %10s' % ('', 'gap us', 'driver us', 'queue us', 'gpu us'))
for name, sub in (('tail (top 10%% gap)', tail), ('rest', rest)):
    if not sub:
        continue
    print('%-22s %10.0f %10.0f %10.0f %10.0f'
          % (name, st.mean(r[4] for r in sub), st.mean(r[1] for r in sub),
             st.mean(r[2] for r in sub), st.mean(r[3] for r in sub)))
print()
print('worst 15 gaps, with the driver time of the SAME buffer:')
print('%-16s %10s %10s %10s %10s' % ('label', 'gap us', 'driver us', 'queue us', 'gpu us'))
for r in gaps[:15]:
    print('%-16s %10.0f %10.0f %10.0f %10.0f' % (r[0][:16], r[4], r[1], r[2], r[3]))

unexplained = [r[4] - r[1] for r in tail]
print()
print('tail: mean gap minus same-buffer driver = %.0f us' % st.mean(unexplained))
print('  ~0 means the idle IS the CPU encoding the next buffer.')
print('  large positive means the CPU was blocked before it even started encoding.')
