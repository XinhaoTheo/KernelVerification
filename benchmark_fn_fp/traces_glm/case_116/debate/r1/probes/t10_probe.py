import torch, json, sys
sys.path.insert(0, "/root/cases/case_116")
import kernel

dev = "cuda"
results = []
cases = [(17,33,257,4),(64,64,512,8),(1,1,1,8),(17,33,33,8),(17,33,33,4),
         (2,2,33,2),(7,5,511,8),(64,64,512,4),(17,33,257,8),(17,33,257,2),(17,33,257,1)]
for m, n, k, sk in cases:
    g = torch.Generator(device="cpu").manual_seed(m*1000+n*10+k+sk)
    a = ((torch.rand((m,k), generator=g)-0.5)*1.0).half().to(dev).clamp(-0.5,0.5)
    b = ((torch.rand((k,n), generator=g)-0.5)*1.0).half().to(dev).clamp(-0.5,0.5)
    bias = ((torch.rand((n,), generator=g)-0.5)*2).float().to(dev).clamp(-1,1)
    ref = kernel.reference(a, b, bias, sk)
    out = kernel.run(a, b, bias, sk)
    err = (out - ref).abs()
    tol = 0.01 + 0.001*ref.abs()
    viol = int((err > tol).sum())
    results.append({"m":m,"n":n,"k":k,"split_k":sk,"max_abs_err":float(err.max()),
                    "tol_min":float(tol.min()),"violations":viol,
                    "exceeds_5x_tol":bool((err > 5*tol).any())})
print(json.dumps({"metric":"per-shape max abs err vs FP64 ref, divisibility edges",
                  "reason":"c2 claims dropped/double-counted K slices at non-divisible K/split_k/32",
                  "cases":results,
                  "any_violation":any(r["violations"]>0 for r in results)}))