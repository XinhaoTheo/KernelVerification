
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_07/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev='cuda'

def build(K, seed, M=32, N=32, gs=8):
    torch.manual_seed(seed)
    G = K//gs
    a = torch.randn(M,K, device=dev, dtype=torch.float32)
    q = torch.randint(0,16,(K,N), device=dev, dtype=torch.int32)
    zst = torch.randint(0,15,(G,N), device=dev, dtype=torch.int32)
    scales = torch.rand(G,N, device=dev, dtype=torch.float32)*0.1+0.01
    g_idx = (torch.arange(K, device=dev)//gs).to(torch.int32)
    b_packed = torch.zeros(K//8, N, device=dev, dtype=torch.int32)
    for row in range(K//8):
        for i in range(8):
            b_packed[row] |= (q[row*8+i] & 0xF) << (4*i)
    zp = torch.zeros(G, N//8, device=dev, dtype=torch.int32)
    for col in range(N//8):
        for i in range(8):
            zp[:,col] |= (zst[:, col*8+i] & 0xF) << (4*i)
    w = (q.double() - (zst.double()[g_idx]+1)) * scales.double()[g_idx]
    return a, b_packed, scales, zp, g_idx, a.double() @ w

res = {}
for K in (32, 48, 64, 24, 40):
    per = []
    for seed in (0, 1, 2):
        try:
            a,bp,sc,zp,gi,ref = build(K, seed)
            c = k.gptq_matmul(a, bp, sc, zp, gi, bits=4)
            torch.cuda.synchronize()
            d = (c.double()-ref).abs()
            mx = float(d.max())
            per.append(dict(seed=seed, maxabs=(None if mx!=mx else mx),
                            is_nan=bool(mx!=mx),
                            nonfinite_out=int((~torch.isfinite(c)).sum()),
                            ref_absmax=float(ref.abs().max()), faulted=False))
        except Exception as ex:
            per.append(dict(seed=seed, faulted=True, error=repr(ex)[:200]))
    res[f"K{K}"] = per

summary = {}
for K, per in res.items():
    bad = sum(1 for p in per if p.get("faulted") or p.get("is_nan") or (p.get("maxabs") or 0) > 1e-3)
    summary[K] = dict(runs=len(per), bad_runs=bad,
                      maxabs_values=[p.get("maxabs") for p in per],
                      nonfinite_counts=[p.get("nonfinite_out") for p in per])
print(json.dumps(dict(detail=res, summary=summary, K_multiples_of_16=[32,48,64], K_tails=[24,40])))
