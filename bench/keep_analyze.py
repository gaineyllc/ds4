import sys, collections
files=sys.argv[1:]
cnt=collections.defaultdict(lambda: collections.Counter()); wsum=collections.defaultdict(lambda: collections.Counter())
for f in files:
    for line in open(f):
        if line.startswith('#'): continue
        l,e,c,w=line.split(); cnt[int(l)][int(e)]+=int(c); wsum[int(l)][int(e)]+=float(w)
Ks=[160,192,224,256,288]
tot_cov={K:[] for K in Ks}
for l in sorted(cnt):
    tot=sum(cnt[l].values()); wt=sum(wsum[l].values())
    ranked=sorted(cnt[l], key=lambda e:-wsum[l][e])
    row=[]
    for K in Ks:
        keep=set(ranked[:K]); cov=sum(cnt[l][e] for e in keep)/tot; wcov=sum(wsum[l][e] for e in keep)/wt
        tot_cov[K].append(cov); row.append(f"K{K}:{cov*100:5.1f}%")
    print(f"layer {l:2d} used={len(cnt[l]):3d} tokens={tot//6:6d} "+" ".join(row))
print("mean selection coverage:", {K: round(100*sum(v)/len(v),1) for K,v in tot_cov.items()})
