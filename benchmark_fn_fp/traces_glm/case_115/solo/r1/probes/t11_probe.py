import itertools, math, random, torch, sys
sys.path.insert(0, "/root/cases/case_115")
import kernel

dev = "cuda"
results = []

def gen(b, qt, d, hk, group, page, lengths, window, seed, c=None):
    g = torch.Generator().manual_seed(seed)
    hq = hk * group
    pages = random.Random(seed).randint(4, 40)
    if c is None:
        c = max((max(lengths) + page - 1)//page + 2, 2)
    q = (2*torch.rand((b, qt, hq, d), generator=g)-1).half()
    k = (2*torch.rand((pages, page, hk, d), generator=g)-1).half()
    v = (2*torch.rand(k.shape, generator=g)-1).half()
    # arbitrary page table with repeats, permutation-ish
    tbl = torch.randint(0, pages, (b, c), generator=g).to(torch.int32)
    # ensure valid prefix
    for i in range(b):
        need = (lengths[i]+page-1)//page
        used = tbl[i, :need].clone()
        # fine as-is (repeats allowed)
        pass
    ln = torch.tensor(lengths, dtype=torch.int32)
    return tuple(t.to(dev) for t in (q,k,v,tbl,ln)) + (window,)

configs = [
    # (b, qt, d, hk, group, page, lengths, window)
    (2,9,64,2,2,16,[97,113],41),        # make_inputs-like
    (1,1,32,1,1,16,[1],0),
    (1,1,32,1,1,16,[1],1),
    (1,1,128,4,4,64,[512],0),
    (1,33,64,1,1,64,[512],0),
    (1,33,64,1,1,64,[512],256),
    (1,33,64,1,1,64,[512],1),
    (2,16,32,2,4,16,[100,300],0),
    (2,16,32,2,4,16,[100,300],7),
    (3,8,64,4,2,32,[64,128,512],0),
    (3,8,64,4,2,32,[64,128,512],63),
    (4,5,32,3,1,16,[500,33,64,129],0),
    (4,5,32,3,1,16,[500,33,64,129],128),
    (1,33,128,2,8,32,[300],5),
    (1,2,64,1,16//1*1,16,[512],0),  # hq=16 group=16? hq/hk in {1,2,4,8}; hk=1, group up to 8, hq<=16
]
# replace bad config with proper hq=8 group=8
configs[-1] = (1,2,64,1,8,16,[512],0)
configs += [
    (1,7,32,1,8,64,[40],9),
    (1,17,64,2,2,16,[200],16),
    (1,9,64,2,2,16,[300],64),
    (2,20,64,2,2,32,[512,512],0),
    (2,20,64,2,2,32,[512,512],100),
    (1,6,32,4,4,16,[512],256),
    (1,6,32,4,4,16,[512],255),
    (1,33,32,1,1,32,[33],0),
    (1,33,32,1,1,32,[33],1),
]

worst = 0.0; worst_cfg = None
for i,(b,qt,d,hk,group,page,lengths,window) in enumerate(configs):
    try:
        inputs = gen(b,qt,d,hk,group,page,lengths,window, seed=1000+i)
        out = kernel.run(*inputs)
        ratio = kernel.error_ratio(out, inputs)
        finite = bool(torch.isfinite(out).all())
        results.append((i, configs[i], ratio, finite))
        if ratio > worst: worst, worst_cfg = ratio, configs[i]
    except Exception as e:
        results.append((i, configs[i], f"ERROR {type(e).__name__}: {e}", None))

import json
print(json.dumps({"metric":"max contract error ratio over sweep","reason":"contract tol is ratio<=1 (abs/ (0.003+0.003*|t|))","results":[[r[0],r[1], (round(r[2],4) if isinstance(r[2],float) else r[2]), r[3]] for r in results], "worst_ratio":worst, "worst_cfg":worst_cfg}))
