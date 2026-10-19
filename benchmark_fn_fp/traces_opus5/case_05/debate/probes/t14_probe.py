
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_06/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev='cuda'
def ref(ns, dA, dt):
    s = torch.zeros(ns.shape[1], device=ns.device, dtype=dt)
    for c in range(ns.shape[0]):
        s = torch.exp(dA[c].to(dt))*s + ns[c].to(dt)
    return s
step = 5e-3
res = {}
cases = {}

torch.manual_seed(0)
nchunks, dim = 32, 64
ns = torch.randn(nchunks, dim, device=dev, dtype=torch.float32)
for name, dA in [("dA_const_-0.1", torch.full((nchunks,), -0.1, device=dev, dtype=torch.float32)),
                 ("dA_uniform_-0.5_0", -0.5*torch.rand(nchunks, device=dev, dtype=torch.float32))]:
    out = k.state_passing_lowbit(ns, dA)
    r64 = ref(ns, dA, torch.float64)
    r32 = ref(ns, dA, torch.float32)
    err = (out.double()-r64).abs()
    fp32err = (r32.double()-r64).abs()
    scaled = out.double()/step
    grid_resid = (scaled - scaled.round()).abs().max().item()
    cases[name] = {
        "max_abs_err": err.max().item(),
        "mean_abs_err": err.mean().item(),
        "max_rel_err": (err/r64.abs().clamp_min(1e-30)).max().item(),
        "fp32_exactscan_max_abs_err": fp32err.max().item(),
        "ratio_err_to_fp32_err": err.max().item()/max(fp32err.max().item(), 1e-30),
        "grid_residual_max": grid_resid,
        "ref_absmax": r64.abs().max().item(),
        "allclose_1e-3": bool(torch.allclose(out.double(), r64, atol=1e-3, rtol=1e-3)),
        "allclose_1e-2": bool(torch.allclose(out.double(), r64, atol=1e-2, rtol=1e-2)),
        "n_entries_err_gt_1e-3": int((err > 1e-3).sum().item()),
        "n_entries": int(err.numel()),
    }
res["shape"] = [nchunks, dim]
res["dtype"] = "float32"
res["quant_step"] = step
res["half_step"] = step/2
res["cases"] = cases
print(json.dumps(res))
