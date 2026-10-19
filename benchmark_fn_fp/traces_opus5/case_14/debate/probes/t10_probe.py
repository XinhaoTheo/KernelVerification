
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_15/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(x, gs):
    y = torch.empty_like(x); n = x.shape[1]
    for s in range(0, n, gs):
        e = min(s+gs, n); g = x[:, s:e]
        am = g.abs().amax(dim=1, keepdim=True)
        sc = torch.where(am == 0, torch.ones_like(am), am/127.0)
        y[:, s:e] = torch.round(g/sc).clamp(-127,127)*sc
    return y

torch.manual_seed(1)
x = torch.randn(4, 40, device="cuda", dtype=torch.float32)
out = k.group_quant_dequant(x, 64)
r = ref(x, 64)
print(json.dumps({
 "shape": list(x.shape), "group_size": 64,
 "out_nonzero_count": int((out != 0).sum().item()),
 "out_total": out.numel(),
 "ref_nonzero_count": int((r != 0).sum().item()),
 "max_abs_err": (out-r).abs().max().item(),
 "ref_max_abs": r.abs().max().item(),
 "allclose_atol1e-3": bool(torch.allclose(out, r, atol=1e-3))
}))
