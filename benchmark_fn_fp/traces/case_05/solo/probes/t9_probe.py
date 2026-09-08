
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_05/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
EPS = k._TIE_EPS
dev='cuda'

def ref(scores, pivot):
    v = scores[scores > pivot]
    if v.numel()==0: return 0
    m = v.min()
    return int(((v-m).abs() < EPS).sum().item())

res=[]
def add(name, s, p):
    s = s.to(dev).float().contiguous()
    try:
        kv = k.count_tied_at_boundary(s, p)
    except Exception as e:
        kv = "ERR:"+repr(e)[:120]
    res.append({"case":name,"N":int(s.numel()),"pivot":float(p),"kernel":kv,"ref":ref(s,float(p))})

torch.manual_seed(1)
# A) power-of-two sizes 1..16384
for e in range(0,15):
    n = 2**e
    s = torch.randint(0,5,(n,)).float()
    add(f"pow2_n{n}_quant", s, 1.0)
    s2 = torch.randn(n)
    add(f"pow2_n{n}_randn", s2, 0.0)
# N=1 special cases
add("n1_above", torch.tensor([3.0]), 0.0)
add("n1_below", torch.tensor([-3.0]), 0.0)

# B) negative / large magnitude tiles
add("neg_all", torch.tensor([-5.0]*10+[-2.0]*7+[-9.0]*(256-17)), -3.0)
add("large_mag", torch.tensor([1e6]*5+[1e6+1.0]*3+[0.0]*(128-8)), 1.0)
add("large_neg_pivot", (torch.randint(0,4,(512,)).float()-1e4), -1e4+1.5)
add("mixed_sign", torch.cat([torch.full((13,),-0.5), torch.full((20,),0.5), torch.full((1024-33,),-7.0)]), -1.0)

# C) near-EPS boundary spacing (kernel's own EPS semantics)
base = 1.0
for d,label in [(0.0,"exact"),(1e-5,"d1e-5"),(5e-4,"d5e-4"),(9.9e-4,"d9.9e-4"),(2e-3,"d2e-3"),(1e-2,"d1e-2")]:
    s = torch.full((256,), 10.0)
    s[:6] = base
    s[6:12] = base + d
    s[12:20] = -1.0
    add(f"spacing_{label}", s, 0.0)

# D) pivot exactly equal to element values (strict >)
s = torch.tensor([2.0]*30 + [3.0]*11 + [5.0]*(512-41))
for p in [2.0, 3.0, 5.0, 1.999999, 2.0000005]:
    add(f"pivot_eq_{p}", s, p)

# E) inf/-inf-free but very wide spread + duplicates of the min above pivot far from others
s = torch.cat([torch.full((9,), 1e-8), torch.full((100,), 1e8), torch.full((128-109,), -1.0)])
add("wide_spread", s, 0.0)

bad=[r for r in res if r["kernel"]!=r["ref"]]
print(json.dumps({"metric":"exact integer count vs reference under kernel EPS=1e-3",
 "n_cases":len(res),"n_mismatch":len(bad),"mismatches":bad,"all":res}))
