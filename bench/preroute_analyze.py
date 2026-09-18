"""What recall can a prerouter actually reach on this data?

Record: uint32 token, uint16 layer, uint16 n_embd, int32 actual[6], half norm[n_embd]
Task:   predict layer L+1's top-6 at token t+1 from layer L's norm at token t.

Baselines to beat, measured in-engine on the same run:
    next-layer-router proxy (Edge0 warm start)  23.1%
    same-layer previous-token (what ds4 does)   26.4%
    bar for a prefetch hint to pay (handoff 8)  ~50%
"""
import sys, struct, numpy as np

N_USED = 6
path = sys.argv[1] if len(sys.argv) > 1 else 'preroute_train1.bin'

raw = open(path, 'rb').read()
# peek the header of record 0 for n_embd
_, _, n_embd = struct.unpack_from('<IHH', raw, 0)
rec = 4 + 2 + 2 + 4 * N_USED + 2 * n_embd
n = len(raw) // rec
print('records %d  n_embd %d  rec %d bytes' % (n, n_embd, rec))

tok = np.zeros(n, np.uint32)
lay = np.zeros(n, np.uint16)
act = np.zeros((n, N_USED), np.int32)
feat = np.zeros((n, n_embd), np.float32)
for i in range(n):
    off = i * rec
    t, l, _ = struct.unpack_from('<IHH', raw, off)
    tok[i], lay[i] = t, l
    act[i] = np.frombuffer(raw, np.int32, N_USED, off + 8)
    feat[i] = np.frombuffer(raw, np.float16, n_embd, off + 8 + 4 * N_USED).astype(np.float32)
del raw

n_expert = int(act.max()) + 1
layers = sorted(set(lay.tolist()))
print('layers %d  experts seen %d  tokens %d' % (len(layers), n_expert, len(set(tok.tolist()))))

# index (token, layer) -> row
idx = {(int(tok[i]), int(lay[i])): i for i in range(n)}

def recall(pred_sets, true_sets):
    hit = tot = 0
    for p, a in zip(pred_sets, true_sets):
        for e in a:
            tot += 1
            if e in p:
                hit += 1
    return 100.0 * hit / tot if tot else 0.0

# ---- baseline 1: static per-layer top-6 by frequency (no features at all)
freq = {l: np.zeros(n_expert, np.int64) for l in layers}
for i in range(n):
    for e in act[i]:
        freq[int(lay[i])][e] += 1
static_top = {l: set(np.argsort(-freq[l])[:N_USED].tolist()) for l in layers}
static_top12 = {l: set(np.argsort(-freq[l])[:12].tolist()) for l in layers}

pairs = []
for (t, l), i in idx.items():
    j = idx.get((t + 1, l + 1))
    if j is not None:
        pairs.append((i, j, l + 1))
print('usable (t,L)->(t+1,L+1) pairs: %d' % len(pairs))

print()
print('static per-layer top-6 by frequency : %.1f%%'
      % recall([static_top[p[2]] for p in pairs], [act[p[1]] for p in pairs]))
print('static per-layer top-12            : %.1f%%'
      % recall([static_top12[p[2]] for p in pairs], [act[p[1]] for p in pairs]))

# ---- baseline 2: same-layer previous token (what ds4 prefetches today)
same = []
for (t, l), i in idx.items():
    j = idx.get((t + 1, l))
    if j is not None:
        same.append((i, j))
print('same-layer previous token          : %.1f%%'
      % recall([set(act[i].tolist()) for i, _ in same], [act[j] for _, j in same]))

# ---- trained: per-layer ridge from norm(t,L) -> multi-hot(t+1,L+1), held out
print()
print('ridge head, per layer, 70/30 split by token:')
tot_hit = tot_all = 0
for L in layers:
    sub = [p for p in pairs if p[2] == L]
    if len(sub) < 60:
        continue
    X = np.stack([feat[i] for i, _, _ in sub])
    Y = np.zeros((len(sub), n_expert), np.float32)
    for r, (_, j, _) in enumerate(sub):
        Y[r, act[j]] = 1.0
    cut = int(len(sub) * 0.7)
    Xtr, Ytr, Xte = X[:cut], Y[:cut], X[cut:]
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    lam = 50.0
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    S = Xte @ W
    top = np.argsort(-S, axis=1)[:, :N_USED]
    for r, (_, j, _) in enumerate(sub[cut:]):
        for e in act[j]:
            tot_all += 1
            if e in top[r]:
                tot_hit += 1
print('  held-out top-6 recall            : %.1f%%'
      % (100.0 * tot_hit / tot_all if tot_all else 0.0))
print()
print('NOTE: ~%d training rows per layer. A 5120->384 head on that is badly'
      % (len(pairs) / max(1, len(layers)) * 0.7))
print('underdetermined; treat the ridge number as a floor, not a ceiling.')
