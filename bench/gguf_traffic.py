import struct, sys, collections
f = open(sys.argv[1], 'rb')
def rd(fmt): 
    s = struct.calcsize(fmt); return struct.unpack('<'+fmt, f.read(s))
def rstr():
    (n,) = rd('Q'); return f.read(n).decode('utf-8', 'replace')
magic = f.read(4); (ver,) = rd('I'); (nt,) = rd('Q'); (nkv,) = rd('Q')
T = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',8:None,9:None,10:'Q',11:'q',12:'d'}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        (et,) = rd('I'); (n,) = rd('Q'); return [rval(et) for _ in range(n)]
    return rd(T[t])[0]
kv = {}
for _ in range(nkv):
    k = rstr(); (t,) = rd('I'); v = rval(t)
    if not isinstance(v, list) or len(v) < 20: kv[k] = v
al = kv.get('general.alignment', 32)
tens = []
for _ in range(nt):
    name = rstr(); (nd,) = rd('I'); dims = rd('Q'*nd); (ty,) = rd('I'); (off,) = rd('Q')
    tens.append((name, dims, ty, off))
hdr_end = f.tell(); data0 = (hdr_end + al - 1)//al*al
import os; fsize = os.path.getsize(sys.argv[1])
# byte size from consecutive offsets (tensors are stored in offset order)
srt = sorted(tens, key=lambda t: t[3])
sizes = {}
for i,(name,dims,ty,off) in enumerate(srt):
    nxt = srt[i+1][3] if i+1 < len(srt) else fsize - data0
    sizes[name] = nxt - off
import math
nel = lambda d: math.prod(d)
bpw = collections.defaultdict(list)
for name,dims,ty,off in tens: bpw[ty].append(sizes[name]/nel(dims))
print("bytes/weight by type:", {t: round(min(v),4) for t,v in bpw.items()})
N_EXP = kv.get('deepseek41.expert_count', 384); TOPK = kv.get('deepseek41.expert_used_count', 6)
print("experts", N_EXP, "top-k", TOPK)
groups = collections.defaultdict(float); total = 0.0
for name,dims,ty,off in tens:
    if not name.startswith('blk.'): 
        key = name
    else:
        key = name.split('.',2)[2]
    b = sizes[name]
    if 'engram' in name: groups['ENGRAM(not per-token)'] += b; continue
    if key.endswith('_exps.weight'): b *= TOPK / N_EXP
    groups[key] += b; total += b
for k,v in sorted(groups.items(), key=lambda x:-x[1])[:16]:
    print(f"{k:40s} {v/1e9:7.3f} GB  {100*v/total if 'ENGRAM' not in k else 0:5.1f}%")
print(f"TOTAL per token (excl. engram) {total/1e9:.3f} GB")
print("attn q_b+output_a+output_b:", round(sum(groups[k] for k in groups if k in ('attn_q_b.weight','attn_output_a.weight','attn_output_b.weight'))/1e9,3), "GB")
types = collections.Counter((n.split('.',2)[2] if n.startswith('blk.') else n, ty) for n,d,ty,o in tens)
print("types:", {k[0]:k[1] for k in types if k[0] in ('attn_q_b.weight','attn_output_a.weight','attn_output_b.weight','ffn_gate_exps.weight','ffn_down_exps.weight','ffn_up_shexp.weight','attn_kv_a_mqa.weight','attn_q_a.weight','output.weight','token_embd.weight')})
