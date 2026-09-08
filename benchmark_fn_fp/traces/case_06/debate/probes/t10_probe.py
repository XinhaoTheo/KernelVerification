
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_06/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev='cuda'; dim=64
def ref(ns, dA, dt):
    s = torch.zeros(ns.shape[1], device=ns.device, dtype=dt)
    for c in range(ns.shape[0]):
        s = torch.exp(dA[c].to(dt))*s + ns[c].to(dt)
    return s
rows=[]
for nc in [1,8,64,512,4096]:
    torch.manual_seed(1)
    ns = torch.randn(nc, dim, device=dev, dtype=torch.float32)
    dA = torch.full((nc,), -1e-3, device=dev, dtype=torch.float32)
    out = k.state_passing_lowbit(ns, dA)
    r64 = ref(ns, dA, torch.float64)
    r32 = ref(ns, dA, torch.float32)
    err=(out.double()-r64).abs()
    rows.append(dict(nchunks=nc,
        max_abs_err=err.max().item(),
        mean_abs_err=err.mean().item(),
        fp32_scan_max_abs_err=(r32.double()-r64).abs().max().item(),
        ref_absmax=r64.abs().max().item(),
        max_rel_err=(err/r64.abs().clamp_min(1e-30)).max().item()))
print(json.dumps(dict(dim=dim, dA_cs=-1e-3, quant_step=5e-3, sweep=rows)))
