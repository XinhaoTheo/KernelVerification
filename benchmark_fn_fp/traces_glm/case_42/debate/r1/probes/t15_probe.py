
import importlib.util, torch, json, math
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_115/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev = "cuda"

# Config: B=1, QT=32 (near-512 allowed positions), Hq=16, Hkv=4, D=128, S=64, P=8, length=512.
b, qt, hq, hk, d, page, pgs, length, window = 1, 32, 16, 4, 128, 64, 8, 512, 0

def make(mode, seed):
    g = torch.Generator().manual_seed(seed)
    if mode == "uniform_cancel":
        # all K equal -> uniform softmax; identity page mapping; V alternates +-u by slot parity
        # -> target = sum_t (-1)^t u /(a+1) = 0 for odd a (pure FP16-rounding stress)
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
        # one logical position (a=0 page? use last position) with K aligned to Q -> dominant logit
        q = (2*torch.rand((b,qt,hq,d), generator=g)-1).half()
        k = (2*torch.rand((pgs,page,hk,d), generator=g)-1).half()
        v = (2*torch.rand((pgs,page,hk,d), generator=g)-1).half()
        table = torch.arange(8, dtype=torch.int32).repeat(b,1)
        # align K of last logical position (t=511 -> page 7 slot 63) to q sign for each head group
        for kv in range(hk):
            k[7,63,kv,:] = q[0,:, (kv*(hq//hk))::(hq//hk)].mean(0).sign().half() * 0.9
            v[7,63,kv,:] = torch.sign(q[0,:, kv*(hq//hk)].mean(0).half()) * 0.9
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
