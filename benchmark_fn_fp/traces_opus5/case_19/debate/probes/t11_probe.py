
import json, importlib.util, torch, torch.nn.functional as F
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_20/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev='cuda'; M,N=4,512; eps=1e-5
x=torch.full((M,N),2.0,device=dev,dtype=torch.float32)
torch.manual_seed(2)
w=torch.randn(N,device=dev); b=torch.randn(N,device=dev)
y=m.layer_norm(x,w,b,eps)
ref=F.layer_norm(x,(N,),w,b,eps)
out={"const_row_max_abs_err":(y-ref).abs().max().item(),
 "const_row_allclose_1e-6":bool(torch.allclose(y,ref,rtol=1e-6,atol=1e-6)),
 "y_equals_bias":bool(torch.allclose(y,b.expand_as(y))),
 "ref_equals_bias":bool(torch.allclose(ref,b.expand_as(ref))),
 "kernel_rstd_at_var0":1.0/eps,"contract_rstd_at_var0":1.0/(eps**0.5)}
# near-constant contrast on same setup
xn=(2.0+1e-4*torch.randn(M,N,device=dev)).contiguous()
yn=m.layer_norm(xn,w,b,eps)
refn=F.layer_norm(xn.double(),(N,),w.double(),b.double(),eps)
out["nearconst_max_abs_err"]=(yn.double()-refn).abs().max().item()
out["nearconst_allclose_1e-2"]=bool(torch.allclose(yn,refn.float(),rtol=1e-2,atol=1e-2))
print(json.dumps(out))
