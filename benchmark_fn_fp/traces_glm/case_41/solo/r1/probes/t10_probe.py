import torch, math, json, sys
sys.path.insert(0, '/root/cases/case_114')
import kernel

dev='cuda'
torch.manual_seed(0)

def build(qt, hq, hk, d, page, window, length, P, C, garbage):
    q = (2*torch.rand((1,qt,hq,d))-1).half().to(dev)
    table = torch.arange(C, dtype=torch.int32).reshape(1,C).to(dev)
    lengths = torch.tensor([length], dtype=torch.int32, device=dev)
    k = (2*torch.rand((P,page,hk,d))-1).half()
    v = (2*torch.rand((P,page,hk,d))-1).half()
    # allowed positions for the single batch: union over rows
    allowed = set()
    for r in range(qt):
        a = length - qt + r
        lo = max(0, a - window + 1) if window else 0
        for t in range(lo, a+1):
            allowed.add((int(table[0, t//page]), t % page))
    # poison every non-allowed slot with garbage
    for p in range(P):
        for s in range(page):
            if (p,s) not in allowed:
                k[p,s] = garbage; v[p,s] = garbage
    return q, k.to(dev), v.to(dev), table, lengths

results=[]
# Case A: minimal, window=1 -> each query attends only to itself; expired slots NaN
q,k,v,t,l = build(1,1,1,64,32,1,64,8,2, float('nan'))
out = kernel.run(q,k,v,t,l,1)
tgt = kernel.reference(q,k,v,t,l,1)
maxerr = float(((out.double()-tgt).abs()/(0.003+0.003*tgt.abs())).max())
results.append(dict(name="window=1,qt=1,len=64,expired=NaN", nan_output=bool((~torch.isfinite(out)).sum()>0), n_nonfinite=int((~torch.isfinite(out)).sum()), maxerr=maxerr))

# Case B: same but expired slots = +1000 (finite garbage, semantically irrelevant)
q,k,v,t,l = build(1,1,1,64,32,1,64,8,2, 1000.0)
out = kernel.run(q,k,v,t,l,1)
tgt = kernel.reference(q,k,v,t,l,1)
maxerr = float(((out.double()-tgt).abs()/(0.003+0.003*tgt.abs())).max())
results.append(dict(name="window=1,expired=1000.0", nan_output=bool((~torch.isfinite(out)).sum()>0), n_nonfinite=int((~torch.isfinite(out)).sum()), maxerr=maxerr))

# Case C: expired slots = 0.0 (benign garbage) -> sanity
q,k,v,t,l = build(1,1,1,64,32,1,64,8,2, 0.0)
out = kernel.run(q,k,v,t,l,1)
tgt = kernel.reference(q,k,v,t,l,1)
maxerr = float(((out.double()-tgt).abs()/(0.003+0.003*tgt.abs())).max())
results.append(dict(name="window=1,expired=0.0 (sanity)", nan_output=bool((~torch.isfinite(out)).sum()>0), n_nonfinite=int((~torch.isfinite(out)).sum()), maxerr=maxerr))

# Case D: window=40, qt=9, length=97, expired NaN
q,k,v,t,l = build(9,4,2,64,16,41,97,8,7, float('nan'))
out = kernel.run(q,k,v,t,l,41)
tgt = kernel.reference(q,k,v,t,l,41)
maxerr = float(((out.double()-tgt).abs()/(0.003+0.003*tgt.abs())).max())
results.append(dict(name="window=41,qt=9,len=97,expired=NaN", nan_output=bool((~torch.isfinite(out)).sum()>0), n_nonfinite=int((~torch.isfinite(out)).sum()), maxerr=maxerr))

print(json.dumps(dict(metric="error ratio + nonfinite count (contract: <=1 and finite)", results=results, verdict_hint="NaN in expired-window slots (semantically irrelevant per contract) corrupts output" if any(r['nan_output'] for r in results) else "no corruption")))
