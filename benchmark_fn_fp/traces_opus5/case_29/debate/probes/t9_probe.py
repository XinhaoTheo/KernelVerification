
import json, sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_29/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

dev='cuda'
def grid_snap(t): return t.to(torch.float8_e4m3fn).to(torch.float32)

res={}
# 1) explicit small-magnitude sweep in [2^-13, 2^-6]
xs=[]
for e in range(-13,-5):
    for m in [1.0,1.1,1.3,1.5,1.7,1.9]:
        xs.append(m*2.0**e)
xs = xs + [-v for v in xs] + [0.001, 1e-4, 2**-10, 2**-11, 2**-9*1.5]
x = torch.tensor(xs, dtype=torch.float32, device=dev)
q = k.fp8_roundtrip(x)
ref = grid_snap(x)
onehot = grid_snap(q)
offgrid = (q != onehot)
res['sweep_n']=x.numel()
res['sweep_offgrid_count']=int(offgrid.sum())
res['sweep_mismatch_vs_ref_count']=int((q!=ref).sum())
# examples
idx=torch.nonzero(offgrid).flatten()[:8].tolist()
res['sweep_examples']=[{'x':float(x[i]),'kernel':float(q[i]),'e4m3_ref':float(ref[i])} for i in idx]
# flush-to-zero check: |x| < 2^-10 must go to 0
tiny = torch.tensor([1e-5,1e-4,2**-12,2**-11,4e-4], dtype=torch.float32, device=dev)
qt = k.fp8_roundtrip(tiny); rt = grid_snap(tiny)
res['tiny_x']=[float(v) for v in tiny]
res['tiny_kernel']=[float(v) for v in qt]
res['tiny_e4m3_ref']=[float(v) for v in rt]
res['tiny_nonzero_but_ref_zero']=int(((rt==0)&(qt!=0)).sum())

# 2) benign randn(4096)
torch.manual_seed(0)
xr = torch.randn(4096, device=dev, dtype=torch.float32)
qr = k.fp8_roundtrip(xr); rr = grid_snap(xr)
sub = (xr.abs()>0)&(xr.abs()<2**-6)
offr = (qr != grid_snap(qr))
res['randn_n']=4096
res['randn_subnormal_region_count']=int(sub.sum())
res['randn_offgrid_count']=int(offr.sum())
res['randn_offgrid_all_in_subnormal_region']=bool((offr & ~sub).sum()==0)
res['randn_mismatch_vs_ref_count']=int((qr!=rr).sum())
res['randn_max_abs_err_vs_ref']=float((qr-rr).abs().max())
res['randn_normal_region_mismatch_count']=int(((qr!=rr)&~sub).sum())
print(json.dumps(res))
