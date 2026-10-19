import importlib.util, torch, json, math
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_115/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev = "cuda"
b, qt, hq, hk, d, page, pgs, length, window = 1, 32, 16, 4, 128, 64, 8, 512, 0
G = hq // hk

def make(mode, seed):
    g = torch.Generator().manual_seed(seed)
    if mode == "uniform_cancel":
        q = (2*torch.rand((b,qt,hq,d), generator=g)-1).half()
        k = torch.full((pgs,page,hk,d), 0.5).half()
        u = (2*torch.randint(0,2,(d,), generator=g).float()-1).half()
        sign = torch.where(torch.arange(page)%2==0, 1.0, -1.0)[:,None,None]
        v = (sign*u[None,None,:]).repeat(pgs,1,hk,1).half()
        table = torch.arange(8, dtype=torch.int32).repeat(b,1)
    elif mode == "sign_pm1":
        q = torch.randint(0,2,(b,qt,hq,d), generator=g).float().mul(2).sub(1).half()
        k = torch.randint(0,2,(pgs,page,hk,d), generator=g).float().mul(2).sub(1).half()
        v = torch.randint(0,2,(pgs,page,hk,d), generator=g).float().mul(2).sub(1).half()
        table = torch.randint(0,pgs,(b,8), generator=g).to(torch.int32)
    elif mode == "dominant":
        q = (2*torch.rand((b,qt,hq,d), generator=g)-1).half()
        k = (2*torch.rand((pgs,page,hk,d), generator=g)-1).half()
        v = (2*torch.rand((pgs,page,hk,d), generator=g)-1).half()
        table = torch.arange(8, dtype=torch.int32).repeat(b,1)
        for kv in range(hk):
            qgrp = q[0, :, kv*G:(kv+1)*G].reshape(-1, d).mean(0)  # [d]
            k[7, 63, kv, :] = (qgrp.sign().half() * 0.9)
            v[7, 63, kv, :] = torch.sign(qgrp).half() * 0.9
    lt = torch.tensor([length]*b, dtype=torch.int32, device=dev)
    return (q.to(dev), k.to(dev), v.to(dev), table.to(dev), lt, window)

results = []
worst = 0.0
for mode in ("uniform_cancel","sign_pm1","dominant"):
    for seed in range(4):
        inputs = make(mode, seed)
        o = kern.run(*inputs)
        tgt = kern.reference(*inputs)
        err = (o.double()-tgt).abs()
        tol = 0.003 + 0.003*tgt.abs()
        ratio = float((err/tol).max())
        worst = max(worst, ratio)
        results.append(dict(mode=mode, seed=seed, max_abs_err=float(err.max()),
                            min_abs_target=float(tgt.abs().min()),
                            max_tolerance_ratio=ratio, violation=bool(ratio>1.0)))
print(json.dumps(dict(metric="max tolerance ratio abs(out-target)/(0.003+0.003|target|) across adversarial FP16-precision stress cases (512 allowed positions, D=128)",
                      config=dict(B=b,QT=qt,Hq=hq,Hkv=hk,D=d,S=page,P=pgs,length=length,window=window),
                      worst_tolerance_ratio=worst, cases=results)))