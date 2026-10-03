
import importlib.util, torch, json, math
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_115/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev = "cuda"

def build(b, qt, hq, hk, d, page, pgs, lengths, window, seed):
    g = torch.Generator().manual_seed(seed)
    q = (2*torch.rand((b,qt,hq,d), generator=g)-1).half()
    k = (2*torch.rand((pgs,page,hk,d), generator=g)-1).half()
    v = (2*torch.rand((pgs,page,hk,d), generator=g)-1).half()
    table = torch.randint(0, pgs, (b, 512//page), generator=g).to(torch.int32)
    lt = torch.tensor(lengths, dtype=torch.int32, device=dev)
    return (q.to(dev), k.to(dev), v.to(dev), table.to(dev), lt, window)

# c3 shapes: partial last query blocks + window not a multiple of TILE=64
combos = []
for (qt,hq,hk,d,page,pgs) in [
    (33,16,2,64,64,8),   # GROUP=8 -> BQ=2, 17 blocks (last partial)
    (17,8,2,64,64,8),    # GROUP=4 -> BQ=4, 5 blocks (last partial)
    (33,16,4,32,16,32),  # GROUP=4 -> BQ=4, D=32, S=16
]:
    for window in (1, 33, 63, 65, 127, 129, 256, 64):
        for lengths in ([512], [200], [97], [33]):
            if any(lengths[li] < qt for li in range(len(lengths))):
                continue
            combos.append(dict(b=1, qt=qt, hq=hq, hk=hk, d=d, page=page, pgs=pgs,
                               lengths=lengths, window=window, seed=len(combos)))
# also window=0 partial-block cases and multi-batch
for window in (0, 41):
    combos.append(dict(b=2, qt=33, hq=16, hk=2, d=64, page=16, pgs=32, lengths=[512,64], window=window, seed=999))

worst = 0.0; worst_case = None; results=[]
for c in combos:
    try:
        inputs = build(**c)
        o = kern.run(*inputs)
        er = kern.error_ratio(o, inputs)
        finite = bool(torch.isfinite(o).all())
        if er > worst:
            worst = er; worst_case = {k:v for k,v in c.items() if k!='seed'}
        results.append(dict(window=c['window'], lengths=c['lengths'], qt=c['qt'],
                            page=c['page'], error_ratio=er, finite=finite))
    except Exception as e:
        results.append(dict(case=str(c), exception=repr(e)))
viol = [r for r in results if isinstance(r.get('error_ratio'), float) and (r['error_ratio']>1.0 or not r.get('finite',True))]
print(json.dumps(dict(
  metric="error_ratio (contract tolerance ratio) per windowed partial-block combo; >1 or non-finite = dropped/mis-scoped allowed position",
  n_cases=len(results), n_violations=len(viol),
  worst_error_ratio=worst, worst_case=worst_case,
  top5=sorted([r for r in results if isinstance(r.get('error_ratio'),float)], key=lambda r:-r['error_ratio'])[:5])))
