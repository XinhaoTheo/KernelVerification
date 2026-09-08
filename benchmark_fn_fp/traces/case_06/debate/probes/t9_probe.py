
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_06/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
dev='cuda'
nchunks, dim = 32, 64
ns = torch.randn(nchunks, dim, device=dev, dtype=torch.float32)
dA = torch.full((nchunks,), -0.1, device=dev, dtype=torch.float32)

def ref(ns, dA, dt):
    s = torch.zeros(ns.shape[1], device=ns.device, dtype=dt)
    for c in range(ns.shape[0]):
        s = torch.exp(dA[c].to(dt))*s + ns[c].to(dt)
    return s

out = k.state_passing_lowbit(ns, dA)
r64 = ref(ns, dA, torch.float64)
r32 = ref(ns, dA, torch.float32)
err = (out.double()-r64).abs()
fp32_ref_err = (r32.double()-r64).abs()
rel = err/r64.abs().clamp_min(1e-30)
step=5e-3
grid_resid = (out.double()/step - (out.double()/step).round()).abs().max().item()
res = dict(
 shape=[nchunks,dim], dtype="float32", dist="new_states~N(0,1), dA_cs=-0.1",
 max_abs_err=err.max().item(), mean_abs_err=err.mean().item(),
 max_rel_err=rel.max().item(),
 fp32_exactscan_max_abs_err=fp32_ref_err.max().item(),
 quant_step=step, half_step=step/2,
 ratio_err_to_fp32refErr=(err.max().item()/max(fp32_ref_err.max().item(),1e-30)),
 kernel_output_on_5e-3_grid_max_residual=grid_resid,
 ref_absmax=r64.abs().max().item(),
 allclose_atol1e-3=bool(torch.allclose(out.double(), r64, atol=1e-3, rtol=1e-3)),
 allclose_atol1e-2=bool(torch.allclose(out.double(), r64, atol=1e-2, rtol=1e-2)),
)
print(json.dumps(res))
