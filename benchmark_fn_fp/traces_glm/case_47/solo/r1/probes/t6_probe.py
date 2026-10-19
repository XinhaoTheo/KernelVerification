import torch, itertools, json, sys
sys.path.insert(0, "/root/cases/case_120")
import kernel as K

dev = "cuda"
def gen(m, kk, n, gs, seed, neg_words=False, worst=False):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    G = (kk+gs-1)//gs
    x = (torch.rand((m,kk), generator=rng)-.5).half().to(dev)
    codes = torch.randint(0,16,(((kk+7)//8)*8, n), generator=rng, dtype=torch.int64)
    qw = torch.zeros(((kk+7)//8, n), dtype=torch.int64)
    for i in range(8): qw |= codes[i::8] << (4*i)
    if neg_words:
        # make some stored words negative signed patterns
        qw = qw | (torch.randint(0,2,qw.shape,generator=rng,dtype=torch.int64) * (1<<31))
    zc = torch.randint(0,16,(G, ((n+7)//8)*8), generator=rng, dtype=torch.int64)
    qz = torch.zeros((G,(n+7)//8), dtype=torch.int64)
    for i in range(8): qz |= zc[:,i::8] << (4*i)
    if neg_words:
        qz = qz | (torch.randint(0,2,qz.shape,generator=rng,dtype=torch.int64) * (1<<31))
    if worst:
        # extreme codes
        codes2 = (torch.rand(((kk+7)//8)*8, n, generator=rng) < .5).long()*15
        qw = torch.zeros(((kk+7)//8,n), dtype=torch.int64)
        for i in range(8): qw |= codes2[i::8] << (4*i)
        zc2 = torch.full((G,((n+7)//8)*8), 15, dtype=torch.int64)  # zero=16
        qz = torch.zeros((G,(n+7)//8), dtype=torch.int64)
        for i in range(8): qz |= zc2[:,i::8] << (4*i)
        sc = torch.full((G,n), 0.03125).half().to(dev)
    else:
        sc = (.002 + .029*torch.rand((G,n), generator=rng)).half().to(dev)
    ids = (torch.arange(kk)//gs)[torch.randperm(kk, generator=rng)].int().to(dev)
    return (x, qw.int().to(dev), sc, qz.int().to(dev), ids, gs)

fails = []
tested = 0
cases = []
for gs in (16,32,64):
    for kk in (1, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65, 127, 128):
        for n in (1, 7, 8, 9, 32, 33, 63, 64, 65, 96):
            for m in (1, 7, 16, 17, 65):
                if kk < gs and gs != 16: continue
                cases.append((m,kk,n,gs))
import random
random.seed(0)
if len(cases) > 400: cases = random.sample(cases, 400)
for (m,kk,n,gs) in cases:
    for seed, neg, worst in ((0,False,False),(1,False,False),(2,True,False),(3,False,True)):
        x,qw,sc,qz,ids,gs_ = gen(m,kk,n,gs,seed,neg_words=neg,worst=worst)
        try:
            out = K.run(x,qw,sc,qz,ids,gs_)
            ref = K.reference(x,qw,sc,qz,ids,gs_)
            err = (out.double()-ref.double()).abs()
            tol = 0.01 + 0.001*ref.double().abs()
            tested += 1
            if bool((err > tol).any()):
                fails.append(dict(m=m,k=kk,n=n,gs=gs,seed=seed,neg=neg,worst=worst,
                                  maxerr=float(err.max()), maxtol=float(tol.min())))
        except Exception as e:
            fails.append(dict(m=m,k=kk,n=n,gs=gs,seed=seed,neg=neg,worst=worst,error=str(e)[:200]))
        if len(fails) >= 5: break
    if len(fails) >= 5: break
print(json.dumps({"tested":tested,"fails":fails}))
