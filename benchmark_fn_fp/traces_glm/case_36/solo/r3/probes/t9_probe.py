import torch, sys
sys.path.insert(0, "/root/cases/case_109")
import kernel as K
torch.cuda.init()
print("dev", torch.cuda.get_device_name(0))

def rand_case(m, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (2*torch.rand((m,n), generator=g)-1).half()
    weight = (0.5+torch.rand((n,), generator=g)).half()
    dy = (2*torch.rand((m,n), generator=g)-1).half()
    rstd = torch.rsqrt(x.float().square().mean(dim=1)+1e-5)
    return [t.cuda() for t in (x, weight, dy, rstd)]

def rms_case(m, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = ((2*torch.rand((m,n), generator=g)-1)*2).half()
    # rows: half tiny RMS, half large RMS
    half = m//2
    rms = x.float().square().mean(dim=1).sqrt()
    target = torch.tensor([0.25]*(m-half)+[4.0]*half)
    x = (x.float()*(target/rms)[:,None]).clamp(-4,4).half()
    rstd = torch.rsqrt(x.float().square().mean(dim=1)+1e-5)
    weight = (0.5+torch.rand(n, generator=g)).half()
    dy = (2*torch.rand((m,n), generator=g)-1).half()
    return [t.cuda() for t in (x, weight, dy, rstd)]

worst = {"dx":0.0,"dw":0.0}
fail = []
def check(tag, m, n, inputs):
    global worst
    snaps = [t.clone() for t in inputs]
    dx, dw = K.run(inputs[0], inputs[1], inputs[2], inputs[3])
    for a,b in zip(inputs,snaps):
        assert torch.equal(a,b), "INPUT MODIFIED"
    r = K.error_ratios((dx,dw), tuple(inputs))
    worst["dx"]=max(worst["dx"],r["dx"]); worst["dw"]=max(worst["dw"],r["dw"])
    assert dx.dtype==torch.float16 and dx.shape==(m,n)
    assert dw.dtype==torch.float32 and dw.shape==(n,)
    assert torch.isfinite(dx.float()).all() and torch.isfinite(dw).all()
    print(tag,m,n,round(r["dx"],4),round(r["dw"],4))
    if r["dx"]>1 or r["dw"]>1: fail.append((tag,m,n,r))

shapes = [(1,16),(1,512),(7,33),(100,127),(511,64),(512,64),(512,100),(1024,255),(768,128),(4096,128),(4096,300),(512,257),(1000,63),(2048,512),(513,16),(2,17),(4096,512)]
for i,(m,n) in enumerate(shapes):
    check("rand", m, n, rand_case(m,n,i))

for i,(m,n) in enumerate([(768,128),(1024,255),(512,64),(4096,300)]):
    check("rms", m, n, rms_case(m,n,1000+i))

for i,(m,n) in enumerate([(1024,64),(512,128),(4096,128)]):
    g=torch.Generator(device="cpu").manual_seed(2000+i)
    x=(2*torch.rand((m,n),generator=g)-1).half()
    weight=(0.5+torch.rand(n,generator=g)).half()
    dy=(torch.randn((m,n),generator=g)*0.01).half()
    rstd=torch.rsqrt(x.float().square().mean(dim=1)+1e-5)
    check("cancel", m, n, [t.cuda() for t in (x,weight,dy,rstd)])

print("WORST", worst)
print("FAILURES", fail)
import json
print(json.dumps({"metric":"error_ratios (<=1 pass), max over shapes/tags","worst":worst,"failures":fail}))
