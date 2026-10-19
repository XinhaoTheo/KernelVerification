
import json, importlib.util, torch
spec=importlib.util.spec_from_file_location("k","/root/cases/case_11/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(x,cos,sin,inter):
    xf=x.float(); c=cos.float()[:,None,:]; s=sin.float()[:,None,:]
    if inter:
        x0=xf[...,0::2]; x1=xf[...,1::2]
        o=torch.stack([x0*c-x1*s, x0*s+x1*c],dim=-1).reshape(xf.shape)
    else:
        d=xf.shape[-1]//2
        x0=xf[...,:d]; x1=xf[...,d:]
        o=torch.cat([x0*c-x1*s, x0*s+x1*c],dim=-1)
    return o.to(x.dtype)

res={}
torch.manual_seed(7)
cases=[(128,8,128,torch.float16),(128,8,128,torch.bfloat16),(1024,16,64,torch.float32),(37,5,80,torch.float16),(2,1,8,torch.float32)]
for S,H,D,dt in cases:
    x=torch.randn(S,H,D,device="cuda",dtype=dt)
    pos=torch.arange(S,device="cuda").float()[:,None]
    inv=(1.0/(10000**(torch.arange(D//2,device="cuda").float()/(D//2))))[None,:]
    ang=pos*inv+0.41
    cos=torch.cos(ang).to(dt).contiguous(); sin=torch.sin(ang).to(dt).contiguous()
    for inter in [False,True]:
        key=f"S{S}_H{H}_D{D}_{str(dt).split('.')[-1]}_inter{int(inter)}"
        try:
            o=m.apply_rotary(x,cos,sin,inter); torch.cuda.synchronize()
            r=ref(x,cos,sin,inter)
            err=(o.float()-r.float()).abs()
            den=r.float().abs().clamp_min(1e-3)
            res[key+"_max_abs_err"]=float(err.max())
            res[key+"_max_rel_err"]=float((err/den).max())
            res[key+"_nbad_2e-2"]=int((err>2e-2).sum())
            res[key+"_finite"]=bool(torch.isfinite(o).all())
        except Exception as e:
            res[key+"_error"]=repr(e)[:300]
print(json.dumps(res))
