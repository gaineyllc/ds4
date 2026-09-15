import sys,re
cyc=[l for l in open(sys.argv[1],errors='replace') if 'dspark cycle' in l]; n=len(cyc)
nod=sum(1 for l in cyc if 'no-draft' in l); par=sum(1 for l in cyc if 'parity' in l); skip=sum(1 for l in cyc if ' skip ' in l)
ver=[l for l in cyc if 'committed=' in l]
agreed=[int(re.search(r'agreed=(\d+)',l).group(1)) for l in cyc if 'agreed=' in l]
com=[int(re.search(r'committed=(\d+)',l).group(1)) for l in ver]
vms=[float(re.search(r'verify=([\d.]+)',l).group(1)) for l in cyc if 'verify=' in l]
dms=[float(re.search(r'draft=([\d.]+)',l).group(1)) for l in cyc if 'draft=' in l]
import collections
print(f'cycles={n} skip={skip} no-draft={nod} parity={par} verified={len(ver)} mean_agreed={sum(agreed)/max(1,len(agreed)):.2f} committed/verified={sum(com)/max(1,len(com)):.2f} verify_ms={sum(vms)/max(1,len(vms)):.0f} draft_ms={sum(dms)/max(1,len(dms)):.1f}')
print('agreed hist', sorted(collections.Counter(agreed).items()))
