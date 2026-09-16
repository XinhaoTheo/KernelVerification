
import json, importlib.util, torch
spec=importlib.util.spec_from_file_location("k","/root/cases/case_11/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def ref(x,cos,sin,inter):
    c=cos[:,None,:]; s=sin[:,None,:]
    if inter:
        x0=x[...,0::2]; x1=x[...,1::2]
        return torch.stack([x0*c-x1*s, x0*s+x1*c],dim=-1).reshape(x.shape)
    d=x.shape[-1]//2
    x0=x[...,:d]; x1=x[...,d:]
    return torch.cat([x0*c-x1*s, x0*s+x1*c],dim=-1)
res={}
torch.manual_seed(2)
for D in [48,96,64]:
    S,H=6,3
    x=torch.randn(S,H,D,device="cuda",dtype=torch.float32)
    ang=torch.randn(S,D//2,device="cuda")*1.3+0.2
    cos=torch.cos(ang).contiguous(); sin=torch.sin(ang).contiguous()
    for inter in [False,True]:
        key=f"D{D}_inter{int(inter)}"
        try:
            o=m.apply_rotary(x,cos,sin,inter); torch.cuda.synchronize()
            r=ref(x,cos,sin,inter)
            err=(o-r).abs()
            res[key+"_max_abs_err"]=float(err.max())
            res[key+"_nbad_1e-4"]=int((err>1e-4).sum())
            res[key+"_finite"]=bool(torch.isfinite(o).all())
        except Exception as e:
            res[key+"_error"]=repr(e)[:400]
print(json.dumps(res))
