
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev="cuda"; out={}

def ref_idx(t,d,iq):
    s=torch.clamp(t-d,min=0.0)*iq
    mx=s.max(dim=1,keepdim=True).values
    return (s==mx).float().argmax(dim=1)

# 1) large sweep of normalized probability rows (fp32)
tot=0; mism=0; zero_mass=0; rows_no_pos=0
torch.manual_seed(7)
for _ in range(20):
    B,V=64,512
    t=torch.softmax(torch.randn(B,V,device=dev),dim=1)
    d=torch.softmax(torch.randn(B,V,device=dev),dim=1)
    iq=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
    k=m.sample_recovered_tokens(t,d,iq); r=ref_idx(t,d,iq)
    resid=torch.clamp(t-d,min=0.0)
    tot+=B; mism+=int((k!=r).sum())
    zero_mass+=int((resid.gather(1,k.view(-1,1)).squeeze(1)==0).sum())
    rows_no_pos+=int(((resid>0).sum(dim=1)==0).sum())
out["norm_fp32_rows"]=tot; out["norm_fp32_mismatch"]=mism
out["norm_fp32_zero_mass_selected"]=zero_mass; out["norm_fp32_rows_without_positive_residual"]=rows_no_pos

# 2) identical distributions (all residuals exactly zero) - tie case
B,V=32,512
d=torch.softmax(torch.randn(B,V,device=dev),dim=1); t=d.clone()
iq=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
k=m.sample_recovered_tokens(t,d,iq); r=ref_idx(t,d,iq)
out["identical_rows"]=B
out["identical_mismatch"]=int((k!=r).sum())
out["identical_kernel_idx_sample"]=k[:8].tolist()
out["identical_ref_idx_sample"]=r[:8].tolist()

# 3) fp16 near-identical normalized rows: rounding can wipe out all positive residuals
B,V=256,512
d16=torch.softmax(torch.randn(B,V,device=dev),dim=1).half()
t16=(d16.float()*(1.0-1e-4)).half()   # slightly below draft everywhere
iq16=(1.0/torch.empty(B,V,device=dev).exponential_(1.0)).half()
k16=m.sample_recovered_tokens(t16,d16,iq16)
resid16=torch.clamp(t16.float()-d16.float(),min=0.0)
r16=ref_idx(t16.float(),d16.float(),iq16.float())
out["fp16_near_rows"]=B
out["fp16_rows_without_positive_residual"]=int(((resid16>0).sum(dim=1)==0).sum())
out["fp16_mismatch"]=int((k16!=r16).sum())
out["fp16_zero_mass_selected"]=int((resid16.gather(1,k16.view(-1,1)).squeeze(1)==0).sum())

# 4) unnormalized/scaled target rows (contract does not state normalization)
B,V=64,512
d=torch.rand(B,V,device=dev)+0.1
t=d*torch.rand(B,1,device=dev)*0.9   # row-scaled strictly below
iq=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
k=m.sample_recovered_tokens(t,d,iq); r=ref_idx(t,d,iq)
resid=torch.clamp(t-d,min=0.0)
out["unnorm_rows"]=B
out["unnorm_rows_without_positive_residual"]=int(((resid>0).sum(dim=1)==0).sum())
out["unnorm_mismatch"]=int((k!=r).sum())
out["unnorm_zero_mass_selected"]=int((resid.gather(1,k.view(-1,1)).squeeze(1)==0).sum())
print(json.dumps(out))
