
import json, importlib.util, torch
spec=importlib.util.spec_from_file_location("k","/root/cases/case_11/kernel.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res={}
S,H,D=8,4,64
torch.manual_seed(1)
x=torch.randn(S,H,D,device="cuda",dtype=torch.float32)
pos=torch.arange(S,device="cuda").float()[:,None]
inv=(1.0/(10000**(torch.arange(D//2,device="cuda").float()/(D//2))))[None,:]
ang=pos*inv+0.37   # asymmetric, sin!=0, cos!=sin, distinct per dim
cos=torch.cos(ang).contiguous(); sin=torch.sin(ang).contiguous()

def ref_inter(x,cos,sin):
    x0=x[...,0::2]; x1=x[...,1::2]
    c=cos[:,None,:]; s=sin[:,None,:]
    o0=x0*c-x1*s; o1=x0*s+x1*c
    o=torch.stack([o0,o1],dim=-1).reshape(x.shape)
    return o
def ref_inter_swapped(x,cos,sin):
    # hypothesised swap: x0<-odd, x1<-even
    x0=x[...,1::2]; x1=x[...,0::2]
    c=cos[:,None,:]; s=sin[:,None,:]
    o0=x0*c-x1*s; o1=x0*s+x1*c
    return torch.stack([o0,o1],dim=-1).reshape(x.shape)
def ref_noninter(x,cos,sin):
    d=x.shape[-1]//2
    x0=x[...,:d]; x1=x[...,d:]
    c=cos[:,None,:]; s=sin[:,None,:]
    return torch.cat([x0*c-x1*s, x0*s+x1*c],dim=-1)

try:
    out=m.apply_rotary(x,cos,sin,True); torch.cuda.synchronize()
    r=ref_inter(x,cos,sin); rs=ref_inter_swapped(x,cos,sin)
    res["max_abs_err_vs_ref_interleaved"]=float((out-r).abs().max())
    res["max_abs_err_vs_swapped_variant"]=float((out-rs).abs().max())
    res["ref_max_abs"]=float(r.abs().max())
    res["mismatch_count_1e-4"]=int(((out-r).abs()>1e-4).sum())
    res["even_lane_max_err"]=float((out[...,0::2]-r[...,0::2]).abs().max())
    res["odd_lane_max_err"]=float((out[...,1::2]-r[...,1::2]).abs().max())
except Exception as e:
    res["interleaved_error"]=repr(e)[:600]
try:
    out2=m.apply_rotary(x,cos,sin,False); torch.cuda.synchronize()
    r2=ref_noninter(x,cos,sin)
    res["max_abs_err_noninterleaved"]=float((out2-r2).abs().max())
except Exception as e:
    res["noninterleaved_error"]=repr(e)[:600]
print(json.dumps(res))
