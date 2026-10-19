import torch, sys
sys.path.insert(0, "/root/cases/case_109")
import kernel as K

def rand_case(m, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (2*torch.rand((m,n), generator=g)-1).half()
    weight = (0.5+torch.rand((n,), generator=g)).half()
    dy = (2*torch.rand((m,n), generator=g)-1).half()
    # keep row RMS in [0.25,4] by construction; values |x|<=4
    rstd = torch.rsqrt(x.float().square().mean(dim=1)+1e-5)
    return (t.cuda() for t in (x, weight, dy, rstd))

def rms_case(m, n, seed):
    # adversarial: rows with tiny RMS (0.25) and large RMS (4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.empty((m,n)).half()
    for i in range(m):
        scale = 0.25 if i%2==0 else 4.0
        x[i] = ((2*torch.rand(n, generator=g)-1)*min(4, scale*4)).half()
    # rescale rows to target RMS exactly-ish, clamp |x|<=4
    rms = x.float().square().mean(dim=1).sqrt()
    target = torch.where(torch.arange(m)%2==0, 0.25, 4.0)
    x = (x.float()*(target/rms)[:,None]).clamp(-4,4).half()
    rms2 = x.float().square().mean(dim=1).sqrt()
    # if clamping moved RMS out of range, retry with random values within bound
    ok = ((rms2>=0.24)&(rms2<=4.05)).all()
    rstd = torch.rsqrt(x.float().square().mean(dim=1)+1e-5)
    weight = (0.5+torch.rand(n, generator=g)).half()
    dy = (2*torch.rand((m,n), generator=g)-1).half()
    return (t.cuda() for t in (x, weight, dy, rstd)), bool(ok)

worst = {"dx":0,"dw":0}
fail = []
# sweep shapes: both branches. Row branch: block>256 (n>256) or m<512; block branch: n<=256 and m>=512
shapes = [(1,16),(1,512),(7,33),(100,127),(511,64),(512,64),(511,64),(512,100),(1024,255),(768,128),(4096,128),(4096,300),(512,257),(1000,63),(2048,512),(513,16),(2,17)]
for i,(m,n) in enumerate(shapes):
    inputs = rand_case(m,n,i)
    snaps = [t.clone() for t in inputs]
    dx, dw = K.run(*inputs)
    for a,b in zip(inputs,snaps):
        assert torch.equal(a,b), "INPUT MODIFIED"
    r = K.error_ratios((dx,dw), inputs)
    worst["dx"]=max(worst["dx"],r["dx"]); worst["dw"]=max(worst["dw"],r["dw"])
    assert dx.dtype==torch.float16 and dx.shape==(m,n) and dw.dtype==torch.float32 and dw.shape==(n,)
    assert torch.isfinite(dx.float()).all() and torch.isfinite(dw).all()
    if r["dx"]>1 or r["dw"]>1:
        fail.append((m,n,r))
    print(m,n,r)

# adversarial rms distribution cases
for i,(m,n) in enumerate([(768,128),(1024,255),(512,64),(4096,300)]):
    inputs, ok = rms_case(m,n,1000+i)
    snaps=[t.clone() for t in inputs]
    dx, dw = K.run(*inputs)
    for a,b in zip(inputs,snaps): assert torch.equal(a,b)
    r = K.error_ratios((dx,dw), inputs)
    worst["dx"]=max(worst["dx"],r["dx"]); worst["dw"]=max(worst["dw"],r["dw"])
    print("rms-case",m,n,ok,r)
    if r["dx"]>1 or r["dw"]>1: fail.append(("rms",m,n,r))

# cancellation-heavy dw: dy signed so column sums nearly cancel
for i,(m,n) in enumerate([(1024,64),(512,128),(4096,128)]):
    g=torch.Generator(device="cpu").manual_seed(2000+i)
    x=(2*torch.rand((m,n),generator=g)-1).half()
    weight=(0.5+torch.rand(n,generator=g)).half()
    dy=torch.randn((m,n),generator=g).half()*0.01
    rstd=torch.rsqrt(x.float().square().mean(dim=1)+1e-5)
    inputs=tuple(t.cuda() for t in (x,weight,dy,rstd))
    dx,dw=K.run(*inputs)
    r=K.error_ratios((dx,dw),inputs)
    worst["dx"]=max(worst["dx"],r["dx"]); worst["dw"]=max(worst["dw"],r["dw"])
    print("cancel",m,n,r)
    if r["dx"]>1 or r["dw"]>1: fail.append(("cancel",m,n,r))

print("WORST", worst)
print("FAILURES", fail)
