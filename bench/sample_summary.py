import sys, re, collections
# Summarize a macOS `sample` call graph: for the main thread, attribute samples to the deepest "interesting" frame.
lines=open(sys.argv[1], errors='replace').read().split('\n')
start=[i for i,l in enumerate(lines) if l.startswith('Call graph')][0]
end=[i for i,l in enumerate(lines) if l.startswith('Total number in stack')][0]
body=lines[start+1:end]
# parse first thread block (main)
stack=[]; leaf=collections.Counter(); total=0
pat=re.compile(r'^([\s+!:|]*)(\d+) (.*)$')
first_depth=None
for l in body:
    m=pat.match(l)
    if not m: continue
    depth=len(m.group(1)); n=int(m.group(2)); name=m.group(3).split('  (in')[0].strip()
    if first_depth is None: first_depth=depth
    if depth==first_depth and stack and total: break  # next thread
    while stack and stack[-1][0]>=depth: stack.pop()
    stack.append((depth,name,n))
    if total==0 and depth==first_depth: total=n
    # a leaf: subsequent line has smaller-or-equal depth -- approximate by recording every node's self time later
    leaf[tuple(s[1] for s in stack[-6:])]+=0
# second pass: compute self samples = n - sum(children)
nodes=[]
for l in body:
    m=pat.match(l)
    if not m: continue
    depth=len(m.group(1)); n=int(m.group(2)); name=m.group(3).split('  (in')[0].strip()
    nodes.append((depth,n,name))
    if len(nodes)>1 and depth==nodes[0][0]: nodes.pop(); break
selfc=collections.Counter()
for i,(d,n,name) in enumerate(nodes):
    child=0
    for d2,n2,_ in nodes[i+1:]:
        if d2<=d: break
        if d2==d+1 or (d2>d and True): pass
    # children are next nodes with depth d+? (sample indents by 2 per level) -> compute as immediate deeper nodes until depth<=d
    j=i+1; 
    while j<len(nodes) and nodes[j][0]>d:
        if nodes[j][0]==nodes[i+1][0] if i+1<len(nodes) else False: child+=nodes[j][1]
        j+=1
    selfc[name]+=n-child
print("main thread samples:", nodes[0][1] if nodes else 0)
for name,c in selfc.most_common(18):
    if c>0: print(f"{c:6d} {100*c/nodes[0][1]:5.1f}%  {name[:110]}")
