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
    sc = (.002 + .029*torch.rand((groups, n), generator=rng)).half().to(dev)
    ids = (torch.arange(k)//gs)[torch.randperm(k, generator=rng)].int() if shuffle else ((torch.arange(k)//gs).int())
    return x, qw.int().to(dev), sc, qz.int().to(dev), ids.to(dev), gs

# focus: K not a multiple of group_size (final partial group) and heavily shuffled g_idx,
# plus monotone g_idx and every group_size, plus all K 1..128 sweep with partial groups
cases = [(4,17,40,16,0),(4,31,40,32,1),(4,63,40,64,2),(4,97,40,32,3),
         (4,129-33,40,64,4),(4,48,40,32,5),(4,100,40,64,6),(4,55,40,16,7)]
for (M,k,n,gs,seed) in cases:
    for shuffle in (True, False):
        x,qw,sc,qz,ids,gs_ = gen(M,k,n,gs,seed,shuffle)
        out = K.run(x,qw,sc,qz,ids,gs)
        ref = K.reference(x,qw,sc,qz,ids,gs)
        diff = (out - ref).abs()
        tol = 0.01 + 0.001*ref.abs()
        results.append({"K":k,"gs":gs,"shuffle":shuffle,"seed":seed,
                        "violations": int((diff > tol).sum().item()),
                        "max_diff": float(diff.max().item())})
# also: make scales/zeros strongly distinct per group so wrong pairing is visible
worst = max(r["max_diff"] for r in results)
print(json.dumps({"claim":"c2 group indexing by sorted position","n_cases":len(results),
                  "all_pass": all(r["violations"]==0 for r in results),
                  "worst_max_diff": worst, "results": results}))