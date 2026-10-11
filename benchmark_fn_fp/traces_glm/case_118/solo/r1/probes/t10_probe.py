import torch, json, random
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_118/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"
fails=[]; n=0; worst=0.0

def check(tag,x,w,ids,r):
    global n, worst
    n+=1
    try:
        out=k.run(x,w,ids,r)
        ref=k.reference(x,w,ids,r)
        assert out.dtype==torch.float32 and out.shape==x.shape[:1]+w.shape[2:], (out.dtype,out.shape)
        assert torch.isfinite(out).all()
        d=(out.double()-ref).abs(); tol=0.01+0.0002*ref.abs()
        worst=max(worst,float(d.max()))
        bad=int((d>tol).sum())
        if bad: fails.append({"case":tag,"n_bad":bad,"max_abs":float(d.max()),
                              "tol_at":float(tol.flatten()[int(d.argmax())])})
    except Exception as e:
        fails.append({"case":tag,"error":repr(e)})

def gen(m,K,N,E,Q,mode,seed):
    g=torch.Generator().manual_seed(seed)
    if mode=="extreme":
        x=(torch.where(torch.rand((m,K),generator=g)<.5,-.5,.5)).half()
        w=(torch.where(torch.rand((E,K,N),generator=g)<.5,-.5,.5)).half()
    else:
        x=(torch.rand((m,K),generator=g)-.5).half()
        w=(torch.rand((E,K,N),generator=g)-.5).half()
    ids=torch.randint(0,E,(m,E),generator=g).argsort(1)[:,:Q].int()
    r=torch.rand((m,Q),generator=g)
    if mode=="zero":
        r=(torch.rand(m,generator=g)<.5).float().unsqueeze(1).expand(m,Q).contiguous()*r
    elif mode=="allzero":
        r=torch.zeros((m,Q))
    else:
        r=r/(r.sum(1,keepdim=True)+.125)
    return x.to(dev),w.to(dev),ids.contiguous().to(dev),r.contiguous().to(dev)

shapes=[(37,33,65,4,2),(1,1,1,1,1),(129,129,129,8,4),(129,1,129,2,1),
        (128,64,32,3,3),(5,17,33,8,2),(129,33,1,8,1),(3,129,129,1,1),
        (16,32,32,8,4),(15,15,15,4,4),(1,129,1,8,4),(129,129,1,8,4)]
for mode in ["rand","zero","allzero","extreme"]:
    for i,(m,K,N,E,Q) in enumerate(shapes):
        check((m,K,N,E,Q,mode),*gen(m,K,N,E,Q,mode,mode=="extreme" and 7 or i))

g=torch.Generator().manual_seed(99)
m,K,N,E,Q=50,64,64,4,2
x=(torch.rand((m,K),generator=g)-.5).half().to(dev)
w=(torch.rand((E,K,N),generator=g)-.5).half().to(dev)
perm=torch.randperm(E-1,generator=g)[:Q]
ids=(1+(perm[None,:]+torch.arange(m)[:,None])%(E-1)).int().to(dev)
r=(torch.rand((m,Q),generator=g)).float().to(dev); r=r/(r.sum(1,keepdim=True)+.125)
check("expert-zero-routes",x,w,ids,r)
ids2=torch.tensor([[0,1]]*m,dtype=torch.int32).to(dev)
check("all-tokens-same-pair",x,w,ids2,r)
ids3=torch.zeros((m,1),dtype=torch.int32).to(dev)
r3=torch.ones((m,1)).to(dev)
check("q1-all-expert0",x,w,ids3,r3)

print(json.dumps({"n_cases":n,"n_fails":len(fails),"worst_abs":worst,"fails":fails}))