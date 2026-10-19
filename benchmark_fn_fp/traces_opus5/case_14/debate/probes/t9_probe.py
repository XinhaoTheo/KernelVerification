
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_15/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(x, gs):
    y = torch.empty_like(x)
    n = x.shape[1]
    for s in range(0, n, gs):
        e = min(s+gs, n)
        g = x[:, s:e]
        am = g.abs().amax(dim=1, keepdim=True)
        sc = torch.where(am == 0, torch.ones_like(am), am/127.0)
        q = torch.round(g/sc).clamp(-127,127)
        y[:, s:e] = q*sc
    return y

torch.manual_seed(0)
x = torch.randn(4, 100, device="cuda", dtype=torch.float32)
out = k.group_quant_dequant(x, 64)
r = ref(x, 64)
head_err = (out[:, :64]-r[:, :64]).abs().max().item()
tail_err = (out[:, 64:]-r[:, 64:]).abs().max().item()
tail_zeros = int((out[:, 64:] == 0).sum().item())
tail_total = out[:, 64:].numel()
tail_x_nonzero = int((x[:, 64:] != 0).sum().item())
print(json.dumps({
 "shape": list(x.shape), "group_size": 64,
 "head_max_abs_err": head_err,
 "tail_max_abs_err": tail_err,
 "tail_exact_zero_count": tail_zeros,
 "tail_total": tail_total,
 "tail_input_nonzero_count": tail_x_nonzero,
 "tail_ref_max_abs": r[:,64:].abs().max().item(),
 "overall_allclose_atol1e-3": bool(torch.allclose(out, r, atol=1e-3))
}))
