
import json, importlib.util, torch, torch.nn.functional as F
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_20/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
dev='cuda'; M,N=8,512; eps=1e-5
res={}
for noise in [1e-4, 1e-3, 1e-2, 1e-1]:
    x = (3.0 + noise*torch.randn(M,N,device=dev,dtype=torch.float32)).contiguous()
    w = torch.randn(N,device=dev,dtype=torch.float32)
    b = torch.randn(N,device=dev,dtype=torch.float32)
    y = m.layer_norm(x,w,b,eps)
    ref = F.layer_norm(x.double(),(N,),w.double(),b.double(),eps).float()
    var = x.double().var(dim=-1,unbiased=False).mean().item()
    denom = ref.abs().clamp_min(1e-6)
    rel = ((y-ref).abs()/denom).max().item()
    res[f"noise_{noise}"]={"row_var_mean":var,
        "kernel_rstd_expected":1.0/(var**0.5+eps),
        "contract_rstd":1.0/((var+eps)**0.5),
        "max_abs_err":(y-ref).abs().max().item(),
        "max_rel_err":rel,
        "allclose_rtol1e-2":bool(torch.allclose(y,ref,rtol=1e-2,atol=1e-2)),
        "finite":bool(torch.isfinite(y).all())}
print(json.dumps(res))
