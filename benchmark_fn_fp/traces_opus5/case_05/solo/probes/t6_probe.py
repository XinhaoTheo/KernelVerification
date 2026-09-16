
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_05/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
EPS = k._TIE_EPS
dev='cuda'

def ref(scores, pivot):
    v = scores[scores > pivot]
    if v.numel()==0: return 0
    m = v.min()
    return int(((v-m).abs() < EPS).sum().item())

results=[]
# 1) explicit small tie groups
for ties in [1,2,3,4,5,7,8,17,100]:
    n=128
    while n < ties+8: n*=2
    s = torch.full((n,), 5.0, device=dev)   # filler above
    s[:ties] = 1.0                          # boundary group
    s[ties:ties+4] = -1.0                   # below pivot
    c = k.count_tied_at_boundary(s, 0.0)
    results.append({"case":f"ties={ties}","kernel":c,"ref":ref(s,0.0)})

# 2) random tiles with quantized values (many ties)
torch.manual_seed(0)
for i in range(8):
    n = 1024
    s = (torch.randint(0,7,(n,),device=dev).float())  # coarse ties
    p = 2.0
    results.append({"case":f"rand_quant{i}","kernel":k.count_tied_at_boundary(s,p),"ref":ref(s,p)})

# 3) continuous randn
for i in range(8):
    n=512
    s = torch.randn(n, device=dev)
    p = float(s.median())
    results.append({"case":f"randn{i}","kernel":k.count_tied_at_boundary(s,p),"ref":ref(s,p)})

# 4) empty above pivot
s = torch.full((64,), -1.0, device=dev)
results.append({"case":"none_above","kernel":k.count_tied_at_boundary(s,0.0),"ref":ref(s,0.0)})

# 5) all above and all tied
s = torch.full((256,), 3.0, device=dev)
results.append({"case":"all_tied_256","kernel":k.count_tied_at_boundary(s,0.0),"ref":ref(s,0.0)})

bad=[r for r in results if r["kernel"]!=r["ref"]]
print(json.dumps({"metric":"exact integer count vs reference","n_cases":len(results),"n_mismatch":len(bad),"mismatches":bad,"all":results}, indent=1))
