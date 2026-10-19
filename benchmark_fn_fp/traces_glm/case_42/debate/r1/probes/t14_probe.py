
import importlib.util, torch, json, traceback
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_115/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev = "cuda"

def build(b, qt, hq, hk, d, page, pgs, lengths, window, seed):
    g = torch.Generator().manual_seed(seed)
    q = (2*torch.rand((b,qt,hq,d), generator=g)-1).half()
    k = torch.full((pgs,page,hk,d), float('nan')).half()
    v = torch.full((pgs,page,hk,d), float('nan')).half()
    table = torch.randint(0, pgs, (b, 512//page), generator=g).to(torch.int32)
    for bi in range(b):
        length = int(lengths[bi]); context = length - qt
        for row in range(qt):
            a = context + row; end = a + 1
            begin = max(0, end - window) if window else 0
            pos = torch.arange(begin, end)
            mapped = table[bi, pos//page].long()
            kk = (2*torch.rand((len(pos),hk,d), generator=g)-1).half()
            vv = (2*torch.rand((len(pos),hk,d), generator=g)-1).half()
            k[mapped, pos%page] = kk
            v[mapped, pos%page] = vv
    q = q.to(dev); k = k.to(dev); v = v.to(dev)
    table = table.to(dev)
    lt = torch.tensor(lengths, dtype=torch.int32, device=dev)
    return (q,k,v,table,lt,window)

cases = [
    dict(b=2, qt=9,  hq=4, hk=2, d=64,  page=16, pgs=16, lengths=[97,113], window=41, seed=1),
    dict(b=2, qt=9,  hq=4, hk=2, d=64,  page=16, pgs=16, lengths=[97,113], window=0,  seed=2),
    dict(b=1, qt=33, hq=16,hk=2, d=64,  page=64, pgs=8,  lengths=[512],    window=256,seed=3),
    dict(b=1, qt=32, hq=16,hk=4, d=128, page=64, pgs=8,  lengths=[512],    window=1,  seed=4),
    dict(b=1, qt=33, hq=16,hk=2, d=64,  page=64, pgs=8,  lengths=[512],    window=1,  seed=5),
    dict(b=3, qt=17, hq=8, hk=2, d=64,  page=16, pgs=32, lengths=[64,65,512],window=33,seed=6),
]
out = []
worst = 0.0
for c in cases:
    try:
        inputs = build(**c)
        snaps = [t.clone() for t in inputs[:5]]
        o = kern.run(*inputs)
        finite = bool(torch.isfinite(o).all())
        immut = all(torch.equal(a,b) for a,b in zip(snaps, inputs[:5]))
        er = kern.error_ratio(o, inputs)
        worst = max(worst, er)
        out.append(dict(case={k:v for k,v in c.items() if k!='seed'},
                        output_finite=finite, inputs_bitwise_unchanged=immut,
                        error_ratio=er, tolerance_violation=bool(er>1.0)))
    except Exception as e:
        out.append(dict(case=c, exception=repr(e)))
print(json.dumps(dict(metric="error_ratio (contract tolerance ratio; >1 means violation) plus output finiteness and input immutability",
                      worst_error_ratio=worst, cases=out)))
