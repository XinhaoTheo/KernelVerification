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
        # dy symmetric so dw cancels: pairs +v/-v
        v = torch.rand((rows//2*2, cols), generator=g)
        dy = torch.cat([v, -v], 0)[:rows].half()
        x = (2*torch.rand((rows,cols),generator=g)-1).half()
        w = (0.5+torch.rand((cols,),generator=g)).half()
    else:  # rms extremes
        # all-|x|=c rows give RMS=c; use c=0.25 and 4 (fp16 exact)
        half = rows//2
        x = torch.ones((rows,cols), generator=g)*0.25
        x[half:] = 4.0
        x = (x*(2*torch.rand((rows,cols),generator=g)-1)).half()
        dy = (2*torch.rand((rows,cols),generator=g)-1).half()
        w = (0.5+torch.rand((cols,),generator=g)).half()
    rms = x.float().square().mean(dim=1)
    rstd = torch.rsqrt(rms + 1e-5)
    return tuple(t.to(dev) for t in (x,w,dy,rstd))

worst={"dx":0.0,"dw":0.0}; bad=[]
i=0
for mode in ["extreme","cancel","rms_extreme"]:
    for m,n in [(1,16),(3,17),(511,100),(512,128),(513,257),(768,300),(4096,512),(2048,64),(999,129)]:
        i+=1
        x,w,dy,rstd = make(m,n,i,mode)
        dx,dw = K.run(x,w,dy,rstd)
        r = K.error_ratios((dx,dw),(x,w,dy,rstd))
        worst["dx"]=max(worst["dx"],r["dx"]); worst["dw"]=max(worst["dw"],r["dw"])
        if r["dx"]>1 or r["dw"]>1 or not torch.isfinite(dx).all() or not torch.isfinite(dw).all():
            bad.append((mode,m,n,r))
        # input preservation
        del x,w,dy,rstd,dx,dw
print(json.dumps({"num_cases":i,"worst":worst,"failures":bad[:10]}))
