
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_06/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev='cuda'
torch.manual_seed(2)
nchunks, dim = 16, 64
ns = 1e-3*torch.randn(nchunks, dim, device=dev, dtype=torch.float32)
dA = torch.full((nchunks,), -0.5, device=dev, dtype=torch.float32)
def ref(ns, dA, dt):
    s = torch.zeros(ns.shape[1], device=ns.device, dtype=dt)
    for c in range(ns.shape[0]):
        s = torch.exp(dA[c].to(dt))*s + ns[c].to(dt)
    return s
out = k.state_passing_lowbit(ns, dA)
r = ref(ns, dA, torch.float64)
err=(out.double()-r).abs()
rel=err/r.abs().clamp_min(1e-30)
zeros_kernel=int((out==0).sum().item())
ref_nonzero=int((r.abs()>0).sum().item())
zeroed_but_ref_nonzero=int(((out==0)&(r.abs()>0)).sum().item())
sign_agree=float((torch.sign(out.double())==torch.sign(r)).float().mean().item())
uniq=torch.unique(out).tolist()
print(json.dumps(dict(
 shape=[nchunks,dim], dist="new_states~1e-3*N(0,1), dA_cs=-0.5",
 quant_step=5e-3,
 kernel_zero_count=zeros_kernel, dim=dim,
 ref_nonzero_count=ref_nonzero,
 zeroed_but_ref_nonzero=zeroed_but_ref_nonzero,
 max_rel_err=rel.max().item(), mean_rel_err=rel.mean().item(),
 max_abs_err=err.max().item(),
 ref_absmax=r.abs().max().item(), ref_absmin=r.abs().min().item(),
 sign_agreement_fraction=sign_agree,
 unique_kernel_values=uniq[:10], n_unique_kernel_values=len(uniq))))
