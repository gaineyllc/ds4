import sys,re
cyc=[l for l in open(sys.argv[1],errors='replace') if 'dspark cycle' in l and 'agreed=' in l]
n=len(cyc); h=n//2
for name,part in (("first half",cyc[:h]),("second half",cyc[h:])):
    ag=[int(re.search(r'agreed=(\d+)',l).group(1)) for l in part]
    print(f"{name}: cycles={len(part)} mean_agreed={sum(ag)/max(1,len(ag)):.2f} P(agreed>=2)={sum(1 for a in ag if a>=2)/max(1,len(ag)):.2f}")
