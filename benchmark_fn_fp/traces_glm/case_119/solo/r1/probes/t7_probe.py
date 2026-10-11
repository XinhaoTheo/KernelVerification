
import torch, sys, json
sys.path.insert(0, "/root/cases/case_119")
import kernel

def make(m, e, q, k, n, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    x = (torch.rand((m, k), generator=g) - .5).half()
    w = (torch.rand((e, k, n), generator=g) - .5).half()
    ids = torch.zeros((m, q), dtype=torch.int32)
    for t in range(m):
        uniq = torch.randperm(e, generator=g)[:q]
        ids[t] = uniq.to(torch.int32)
    route = torch.rand((m, q), generator=g)
    route = route / (route.sum(dim=1, keepdim=True) + .125)
    return x.cuda(), w.cuda(), ids.cuda().contiguous(), route.float().cuda()

results = []
cases = [
    (37, 4, 2, 33, 65, 0),
    (65, 4, 2, 33, 65, 1),
    (50, 4, 2, 17, 97, 2),
    (100, 4, 2, 33, 65, 3),
    (129, 4, 4, 33, 65, 4),
    (70, 3, 3, 33, 65, 5),
    (44, 4, 2, 33, 65, 6),
    (21, 5, 3, 33, 65, 7),
    (33, 6, 2, 33, 65, 8),
]
worst = 0.0
for (m,e,q,k,n,seed) in cases:
    x,w,ids,rw = make(m,e,q,k,n,seed)
    c = torch.bincount(ids.reshape(-1).cpu().long(), minlength=e)
    padded = ((c + 15)//16)*16
    nblocks = int((padded.sum()//16).item())
    out = kernel.run(x,w,ids,rw)
    ref = kernel.reference(x,w,ids,rw)
    err = (out.double() - ref).abs()
    tol = 0.01 + 0.0002*ref.abs()
    viol = int((err > tol).sum().item())
    maxerr = err.max().item()
    worst = max(worst, maxerr)
    bad_cols = (err > tol).any(dim=0).nonzero().flatten().tolist()
    results.append(dict(m=m,e=e,q=q,k=k,n=n,blocks=nblocks,
                        blocks_mod4=nblocks%4,
                        violations=viol, max_abs_err=maxerr,
                        bad_col_indices=bad_cols[:40], n_bad_cols=len(bad_cols),
                        n=int(n)))
    print(results[-1])

print("WORST", worst)
print(json.dumps({"metric": "elementwise |actual-target| vs 0.01+0.0002|target|",
                  "worst_max_abs_err": worst, "results": results}))
