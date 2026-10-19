
import torch, math, random, json, sys
sys.path.insert(0, '/root/cases/case_114')
import kernel

random.seed(123)
torch.manual_seed(7)
dev = 'cuda'
worst = 0.0; worst_case = None; fails = []

def gen_case(b, qt, hq, hk, d, page, window, length, P, C, put_garbage=True):
    q = (2*torch.rand((b,qt,hq,d))-1).half().to(dev)
    # build table with repeated/permuted pages
    used_pages = torch.randperm(P)[:min(P, math.ceil(length/page))]
    # allow repeats sometimes
    if random.random() < 0.5 and len(used_pages) >= 2:
        used_pages[0] = used_pages[-1]
    table = torch.randint(0, P, (b, C), dtype=torch.int32)  # garbage tail columns
    for i in range(b):
        n = math.ceil(length/page)
        table[i,:n] = used_pages[:n]
    lengths = torch.tensor([length]*b, dtype=torch.int32, device=dev)
    k = (2*torch.rand((P,page,hk,d))-1).half().to(dev)
    v = (2*torch.rand((P,page,hk,d))-1).half().to(dev)
    if put_garbage:
        # poison pages never referenced by any allowed position
        refd = set(table[i,:math.ceil(length/page)].tolist() for i in range(b))
        # simpler: poison pages >= some id not in table's live portion
        live = set()
        for i in range(b):
            live.update(table[i,:math.ceil(length/page)].tolist())
        dead = [p for p in range(P) if p not in live]
        # also poison slots beyond ceil(length/page)*page within live pages
        k = k.cpu(); v = v.cpu()
        import numpy as np
        for p in dead[:max(1,len(dead)//2)]:
            k[p] = float('nan'); v[p] = float('nan')
        for i in range(b):
            n = math.ceil(length/page)
            tail = n*page - length
            if tail > 0:
                for p in table[i,:n].tolist():
                    k[p, page-tail:] = float('nan'); v[p, page-tail:] = float('nan')
        k = k.to(dev); v = v.to(dev)
    return (q, k, v, table.to(dev), lengths), window

cases = []
cases.append(gen_case(2, 9, 4, 2, 64, 16, 41, 97, 16, 8))
cases.append(gen_case(1, 33, 8, 1, 32, 64, 0, 300, 40, 8))
cases.append(gen_case(3, 5, 16, 4, 128, 16, 256, 200, 30, 16))
cases.append(gen_case(1, 1, 1, 1, 64, 32, 1, 64, 8, 2))
cases.append(gen_case(4, 33, 16, 2, 64, 32, 7, 150, 24, 8))
cases.append(gen_case(2, 33, 2, 2, 32, 64, 3, 33, 10, 2))  # window smaller than Q span
cases.append(gen_case(1, 2, 4, 1, 32, 16, 256, 512, 64, 32))
cases.append(gen_case(2, 8, 8, 4, 128, 64, 100, 500, 40, 8))
cases.append(gen_case(1, 5, 12, 3, 64, 16, 0, 90, 12, 6))
cases.append(gen_case(3, 33, 3, 3, 64, 32, 13, 512, 40, 16))

for idx, (inp, w) in enumerate(cases):
    q,k,v,t,l = inp
    snap = [x.clone() for x in (q,k,v,t,l)]
    out = kernel.run(q,k,v,t,l,w)
    tgt = kernel.reference(q,k,v,t,l,w)
    err = ((out.double()-tgt).abs() / (0.003+0.003*tgt.abs()))
    maxerr = float(err.max())
    finite = bool(torch.isfinite(out).all())
    unchanged = all(torch.equal(a,b) for a,b in zip((q,k,v,t,l), snap))
    info = dict(case=idx, shape=list(q.shape)+[list(k.shape)], w=w, lens=[int(x) for x in l], maxerr=round(maxerr,4), finite=finite, unchanged=unchanged)
    if maxerr > worst: worst, worst_case = maxerr, info
    if maxerr > 1.0 or not finite or not unchanged:
        fails.append(info)
    print(json.dumps(info))

print(json.dumps(dict(metric="max relative error ratio (contract: <=1)", worst=round(worst,4), worst_case=worst_case, fails=fails, n=len(cases))))
