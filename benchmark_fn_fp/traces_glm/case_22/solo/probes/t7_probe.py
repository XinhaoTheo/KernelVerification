import torch, json
from kernel import splitk_dot
torch.manual_seed(0)
res=[]
for K in [0, 1, 7, 255, 256, 257, 511, 512, 513, 4096, 100000, 1000003]:
    a = torch.randn(K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, device='cuda', dtype=torch.float32)
    if K == 0:
        a = torch.zeros(0, device='cuda'); b = torch.zeros(0, device='cuda')
    got = splitk_dot(a, b)
    ref64 = (a.double() * b.double()).sum().item()
    got = got.item()
    rel = abs(got - ref64) / max(abs(ref64), 1e-30)
    res.append({"K": K, "got": got, "ref64": ref64, "abs_err": abs(got-ref64), "rel_err": rel})
# also a magnitude-heavy case: values ~1e-3 and ~1e3
K = 100000
a = (torch.randn(K, device='cuda')*1e-3).float(); b=(torch.randn(K, device='cuda')*1e3).float()
got = splitk_dot(a,b).item(); ref64=(a.double()*b.double()).sum().item()
res.append({"K": K, "dist": "scaled 1e-3/1e3", "got": got, "ref64": ref64, "abs_err": abs(got-ref64), "rel_err": abs(got-ref64)/max(abs(ref64),1e-30)})
for r in res: print(json.dumps(r))
print("max_rel_err", max(r["rel_err"] for r in res))
