import sys, collections
# usage: make_keep.py OUT K [--equalize] usage1.txt [usage2.txt ...]
out=sys.argv[1]; K=int(sys.argv[2]); files=[a for a in sys.argv[3:] if not a.startswith('--')]; eq='--equalize' in sys.argv
cnt=collections.defaultdict(collections.Counter); wsum=collections.defaultdict(collections.Counter)
for f in files:
    for line in open(f):
        if line.startswith('#'): continue
        l,e,c,w=line.split(); cnt[int(l)][int(e)]+=int(c); wsum[int(l)][int(e)]+=float(w)
layers=sorted(cnt); n_exp=384
# score = weight-sum (REAP-lite: frequency x mean router weight)
ranked={l: sorted(range(n_exp), key=lambda e:-wsum[l][e]) for l in layers}
if not eq:
    keep={l: ranked[l][:K] for l in layers}
else:
    # equalize coverage: give layers with flatter distributions more experts, total = K*len(layers)
    budget=K*len(layers); keep={l: ranked[l][:6] for l in layers}
    # greedy: repeatedly add the expert with the highest marginal weight share to the layer where it helps most
    import heapq
    heap=[]
    for l in layers:
        tot=sum(wsum[l].values()) or 1.0
        for i in range(6, n_exp): heapq.heappush(heap, (-wsum[l][ranked[l][i]]/tot, l, i))
    used=sum(len(v) for v in keep.values())
    while used<budget and heap:
        _,l,i=heapq.heappop(heap); keep[l].append(ranked[l][i]); used+=1
with open(out,'w') as o:
    o.write(f"# expert keep-list: K={K} equalize={eq} from {files}\n")
    for l in layers:
        ks=sorted(keep[l]); tot=sum(wsum[l].values()) or 1
        cov=sum(wsum[l][e] for e in ks)/tot
        o.write(f"# layer {l}: {len(ks)} experts, weight coverage {cov*100:.1f}%\n{l}: "+" ".join(map(str,ks))+"\n")
print("wrote", out, "layers", len(layers), "total experts", sum(len(v) for v in keep.values()))
