import sys, json, torch
sys.path.insert(0, "/root/cases/case_120")
import kernel as K

dev = "cuda" if torch.cuda.is_available() else "cpu"
results = []
def gen(M, k, n, gs, seed, shuffle=True):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    groups = (k + gs - 1)//gs
    x = (torch.rand((M, k), generator=rng) - .5).half().to(dev)
    codes = torch.randint(0, 16, (((k+7)//8)*8, n), generator=rng, dtype=torch.int64)
    qw = torch.zeros(((k+7)//8, n), dtype=torch.int64)
    for i in range(8): qw |= codes[i::8] << (4*i)
    zc = torch.randint(0, 16, (groups, ((n+7)//8)*8), generator=rng, dtype=torch.int64)
    qz = torch.zeros((groups, (n+7)//8), dtype=torch.int64)
    for i in range(8): qz |= zc[:, i::8] << (4*i)
    # adversarial: also make some signed words negative
    qw[0, :] |= (torch.randint(0, 2, (n,), generator=rng) * (2**31))  # arbitrary high bits
    sc = (.002 + .029*torch.rand((groups, n), generator=rng)).half().to(dev)
    ids = (torch.arange(k)//gs)[torch.randperm(k, generator=rng)].int() if shuffle else ((torch.arange(k)//gs).int())
    return x, qw.int().to(dev), sc, qz.int().to(dev), ids.to(dev), gs

cases = [(7,64,65,16,0),(1,128,96,64,1),(65,63,33,32,2),(65,128,64,16,3),
         (33,17,1,16,4),(65,127,95,64,5),(5,16,96,16,6),(17,33,65,32,7),
         (65,129%129 if False else 96,96,64,8),(9,79,49,16,9)]
for (M,k,n,gs,seed) in cases:
    x,qw,sc,qz,ids,gs_ = gen(M,k,n,gs,seed)
    out = K.run(x,qw,sc,qz,ids,gs)
    ref = K.reference(x,qw,sc,qz,ids,gs)
    diff = (out - ref).abs()
    tol = 0.01 + 0.001*ref.abs()
    viol = int((diff > tol).sum().item())
    results.append({"M":M,"K":k,"N":n,"gs":gs,"seed":seed,
                    "max_abs_diff": float(diff.max().item()),
                    "tol_violations": viol,
                    "max_target_abs": float(ref.abs().max().item())})
print(json.dumps({"claim":"c1 layout inversion end-to-end","n_cases":len(results),
                  "all_pass": all(r["tol_violations"]==0 for r in results),
                  "max_diff_overall": max(r["max_abs_diff"] for r in results),
                  "results": results}))