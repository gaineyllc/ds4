import sys, collections
B=[]; E=collections.defaultdict(list)
for line in open(sys.argv[1]):
    if line.startswith('B '):
        _,seq,n,s,e = line.split()[:5]; B.append((int(seq),int(n),int(s),int(e)))
    elif line.startswith('E '):
        p=line.split(); E[int(p[1])].append((int(p[3]),int(p[4]),float(p[5]),float(p[6]),p[-1]))
B.sort(key=lambda b:b[2])
# take the last 20 tokens = last 20*40 cbs? find cb count per token: assume 2 cbs per layer -> 80/token
N=len(B); tail=B[-80*20:]
busy=sum(e-s for _,_,s,e in tail)/1e6
span=(tail[-1][3]-tail[0][2])/1e6
gaps=[(tail[i+1][2]-tail[i][3])/1e6 for i in range(len(tail)-1)]
print(f"cbs total {N}; tail {len(tail)} cbs; span {span:.1f} ms; gpu busy {busy:.1f} ms ({100*busy/span:.0f}%); idle {span-busy:.1f} ms")
print(f"per token (span/20): {span/20:.1f} ms; busy {busy/20:.1f}; idle {(span-busy)/20:.1f}")
import statistics
gs=sorted(gaps); print("gap ms: median %.3f p90 %.3f p99 %.3f max %.3f, sum %.1f"%(statistics.median(gs),gs[int(.9*len(gs))],gs[int(.99*len(gs))],gs[-1],sum(gs)))
# biggest gaps with the kernel before/after
big=sorted(range(len(gaps)),key=lambda i:-gaps[i])[:12]
for i in big:
    b0=tail[i]; b1=tail[i+1]
    k0=E[b0[0]][-1][4] if E[b0[0]] else '?'; k1=E[b1[0]][0][4] if E[b1[0]] else '?'
    print(f"  gap {gaps[i]:7.3f} ms after cb{b0[0]} (last {k0[:40]}) before cb{b1[0]} (first {k1[:40]}) n_enc {b0[1]}->{b1[1]}")
# kernel time share in the tail
kt=collections.defaultdict(float); kn=collections.Counter()
for _,_,_,_ in [(0,0,0,0)]: pass
seqs=set(b[0] for b in tail)
for s in seqs:
    for st,en,du,ga,k in E[s]: kt[k]+=du/1000; kn[k]+=1
tot=sum(kt.values())
print(f"kernel time sum {tot:.1f} ms over 20 tokens = {tot/20:.1f} ms/token; dispatches/token {sum(kn.values())/20:.0f}")
for k,v in sorted(kt.items(),key=lambda x:-x[1])[:14]: print(f"  {v/20:6.2f} ms/tok {kn[k]/20:6.1f}/tok {v/kn[k]*1000:7.1f} us  {k}")
# intra-cb encoder gaps
ig=[]
for s in seqs:
    es=sorted(E[s]); 
    for i in range(len(es)-1): ig.append((es[i+1][0]-es[i][1])/1e3)
print(f"intra-cb encoder gaps: sum {sum(ig)/20:.2f} ms/token, median {statistics.median(ig):.1f} us, n/token {len(ig)/20:.0f}")
