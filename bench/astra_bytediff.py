"""Byte-diff the generated text of two ab16k runs, ignoring timing/stat lines.

Timing on this box drifts +-12% run to run for identical binaries, so the
generated text is the only correctness signal that means anything.
"""
import sys

NL = chr(10)
SKIP = ('ds4: prefill:', 'processing ')
DROP_IF_IN = ('zero-copy', 'windows:', 'draft-token expert hints',
              'per layer per pass', 'rows=', 'gather failed', 'memory detail')


def body(path):
    text = open(path, errors='replace').read()
    i = text.find('processing 5723 input tokens')
    if i >= 0:
        text = text[i:]
    keep = []
    for line in text.split(NL):
        if line.startswith(SKIP):
            continue
        if any(s in line for s in DROP_IF_IN):
            continue
        keep.append(line)
    return NL.join(keep).strip()


a_path, b_path = sys.argv[1], sys.argv[2]
a, b = body(a_path), body(b_path)
print('a', a_path, len(a), 'chars')
print('b', b_path, len(b), 'chars')
if a == b:
    print('RESULT: IDENTICAL')
    sys.exit(0)
print('RESULT: DIFFER')
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        print('first diff at char', i)
        print('a:', repr(a[max(0, i - 80):i + 80]))
        print('b:', repr(b[max(0, i - 80):i + 80]))
        break
sys.exit(1)
