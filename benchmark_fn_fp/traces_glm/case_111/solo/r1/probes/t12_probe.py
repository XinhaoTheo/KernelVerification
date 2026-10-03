import sys, json, torch
sys.path.insert(0, "/root/cases/case_111")
import kernel as K
dev="cuda"
def make(rows, cols, seed, mode):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "extreme":
        x = (4*(2*torch.rand((rows,cols),generator=g)-1)).half()
        dy = (4*(2*torch.rand((rows,cols),generator=g)-1)).half()
        w = (2*(2*torch.rand((cols,),generator=g)-1)).half()
    elif mode == "cancel":
        half = (rows+1)//2
        v = torch.rand((half, cols), generator=g)
        dy = torch.cat([v, -v], 0)[:rows].half()
        x = (2*torch.rand((rows,cols),generator=g)-1).half()
        w = (0.5+torch.rand((cols,),generator=g)).half()
    else:
        half = rows//2
        x = torch.ones((rows,cols))*0.25
        x[half:] = 4.0
        x = (x*(2*torch.rand((rows,cols),generator=g)-1)).half()
        dy = (2*torch.rand((rows,cols),generator=g)-1).half()
        w = (0.5+torch.rand((cols,),generator=g)).half()
    rms = x.float().square().mean(dim=1)
    rstd = torch.rsqrt(rms + 1e-5)
    return tuple(t.to(dev) for t in (x,w,dy,rstd))

worst={}; per_mode={}
i=0; bad=[]
for mode in ["extreme","cancel","rms_extreme"]:
    pm={"dx":0.0,"dw":0.0}
    for m,n in [(1,16),(3,17),(511,100),(512,128),(513,257),(768,300),(4096,512),(2048,64),(999,129),(4096,512)]:
        i+=1
        x,w,dy,rstd = make(m,n,i,mode)
        xc,wc,dyc,rc = [t.clone() for t in (x,w,dy,rstd)]
        dx,dw = K.run(x,w,dy,rstd)
        r = K.error_ratios((dx,dw),(x,w,dy,rstd))
        pm["dx"]=max(pm["dx"],r["dx"]); pm["dw"]=max(pm["dw"],r["dw"])
        if r["dx"]>1 or r["dw"]>1 or not torch.isfinite(dx).all() or not torch.isfinite(dw).all():
            bad.append((mode,m,n,r))
        assert torch.equal(x,xc) and torch.equal(w,wc) and torch.equal(dy,dyc) and torch.equal(rstd,rc), "input modified"
    per_mode[mode]=pm
    worst["dx"]=max(worst.get("dx",0),pm["dx"]); worst["dw"]=max(worst.get("dw",0),pm["dw"])
print(json.dumps({"num_cases":i,"worst":worst,"per_mode":per_mode,"failures":bad[:10],"inputs_preserved":True}))
