
import torch, json, itertools, random
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_118/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
random.seed(1)
fails = []
n_cases = 0

def gen(m, K, N, E, Q, mode, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand((m,K), generator=g)-.5).half()
    w = (torch.rand((E,K,N), generator=g)-.5).half()
    if mode=="skew":
        ids = torch.full((m,Q), E-1, dtype=torch.int32)
        for i in range(m):
            s = random.sample(range(E), Q); ids[i]=torch.tensor(s, dtype=torch.int32)
    elif mode=="single_expert":
        ids = torch.full((m,Q), 0, dtype=torch.int32)
        for i in range(m):
            s = random.sample(range(E), Q); ids[i]=torch.tensor(s, dtype=torch.int32)
        ids[:] = ids[:, :1]  # placeholder, fixed below
        # all tokens pick same Q experts
        s = sorted(random.sample(range(E), Q)); ids = torch.tensor([s]*m, dtype=torch.int32)
    else:
        ids = torch.randint(0, E, (m, max(E,1)), generator=g).argsort(dim=1)[:, :Q].int()
    if Q==1: ids = torch.zeros((m,1), dtype=torch.int32)
    r = torch.rand((m,Q), generator=g)
    if mode=="zero":
        r = (torch.rand(m, generator=g) < .5).float()[:,None]
    r = (r/(r.sum(1,keepdim=True)+.125)).float()
    return x.to(dev), w.to(dev), ids.contiguous().to(dev), r.contiguous().to(dev)

cases = []
for mode in ["rand","skew","zero"]:
    cases += [(37,33,65,4,2,mode),(1,1,1,1,1,mode),(129,129,129,8,4,mode),(129,1,129,2,1,mode),
              (128,64,32,3,3,mode),(5,17,33,8,2,mode),(129,33,1,8,1,mode),(3,129,129,1,1,mode)]
# also an expert receiving zero routes: force ids to exclude expert 0
for seed,(m,K,N,E,Q,mode) in enumerate(cases):
    x,w,ids,r = gen(m,K,N,E,Q,mode,seed)
    try:
        out = k.run(x,w,ids,r)
        ref = k.reference(x,w,ids,r)
        diff = (out.double()-ref).abs()
        tol = 0.01 + 0.0002*ref.abs()
        bad = (diff > tol)
        n_cases += 1
        if bad.any():
            fails.append({"case":(m,K,N,E,Q,mode),"n_bad":int(bad.sum()),
                "max_abs":float(diff.max()),"max_rel":float((diff/(ref.abs()+1e-9)).max()),
                "tol_at_max":float(tol.flatten()[int(diff.argmax())])})
    except Exception as e:
        n_cases += 1
        fails.append({"case":(m,K,N,E,Q,mode),"error":repr(e)})

# expert with zero routes: ids all in {1..E-1}
g=torch.Generator().manual_seed(99)
E,Q,m,K,N=4,2,50,64,64
x=(torch.rand((m,K),generator=g)-.5).half().to(dev)
w=(torch.rand((E,K,N),generator=g)-.5).half().to(dev)
ids=torch.randint(1,E,(m,)).int()
ids=torch.stack([ids,(ids+1)%E],1).to(dev)  # may not be distinct; fix
ids = torch.tensor([[ (i%(E-1))+1, (i%(E-1))+1 ] for i in range(m)])  # need distinct
ids2=torch.randint(0,E,(m,E),generator=g).argsort(1)[:, :Q].int().to(dev)
# build distinct ids avoiding expert 0: rotate among 1..E-1
perm = torch.randperm(E-1, generator=g)[:Q]
ids = (1 + (perm[None,:] + torch.arange(m)[:,None].long()) % (E-1)).int().to(dev)
r=(torch.rand((m,Q),generator=g)*0.9).float().to(dev)
r = r/(r.sum(1,keepdim=True)+.125)
try:
    out=k.run(x,w,ids,r); ref=k.reference(x,w,ids,r)
    diff=(out.double()-ref).abs(); tol=0.01+0.0002*ref.abs()
    n_cases+=1
    if (diff>tol).any(): fails.append({"case":"zero-route-expert","max_abs":float(diff.max())})
except Exception as e:
    n_cases+=1; fails.append({"case":"zero-route-expert","error":repr(e)})

print(json.dumps({"n_cases":n_cases,"n_fails":len(fails),"fails":fails}))
