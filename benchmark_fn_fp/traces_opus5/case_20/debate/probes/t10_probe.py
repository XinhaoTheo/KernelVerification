
import json, importlib.util, torch, torch.nn.functional as F
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_20/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(1)
dev='cuda'; M,N=64,512; eps=1e-5
x=torch.randn(M,N,device=dev,dtype=torch.float32)
w=torch.randn(N,device=dev); b=torch.randn(N,device=dev)
y=m.layer_norm(x,w,b,eps)
ref=F.layer_norm(x.double(),(N,),w.double(),b.double(),eps).float()
refd=F.layer_norm(x.double(),(N,),w.double(),b.double(),eps)
var=x.double().var(dim=-1,unbiased=False)
rstd_k=1.0/(var.sqrt()+eps); rstd_c=1.0/(var+eps).sqrt()
rel_rstd=((rstd_k-rstd_c).abs()/rstd_c).max().item()
denom=refd.abs().clamp_min(1e-3)
rel_y=((y.double()-refd).abs()/denom).max().item()
out={"max_rel_rstd_err":rel_rstd,"eps_over_2":eps/2,
 "max_abs_err_y":(y.double()-refd).abs().max().item(),
 "max_rel_err_y_denom1e-3":rel_y,
 "allclose_rtol1e-3":bool(torch.allclose(y,ref,rtol=1e-3,atol=1e-3)),
 "allclose_rtol1e-5":bool(torch.allclose(y,ref,rtol=1e-5,atol=1e-5)),
 "allclose_rtol1e-6":bool(torch.allclose(y,ref,rtol=1e-6,atol=1e-6)),
 "var_mean":var.mean().item()}
# larger eps case
for e in [1e-1, 1.0]:
    y2=m.layer_norm(x,w,b,e)
    r2=F.layer_norm(x.double(),(N,),w.double(),b.double(),e)
    out[f"eps_{e}_max_abs_err"]=(y2.double()-r2).abs().max().item()
    out[f"eps_{e}_allclose_1e-2"]=bool(torch.allclose(y2,r2.float(),rtol=1e-2,atol=1e-2))
print(json.dumps(out))
