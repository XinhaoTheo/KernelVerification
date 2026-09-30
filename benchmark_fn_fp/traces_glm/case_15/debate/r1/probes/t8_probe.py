import torch, sys
sys.path.insert(0, "/root/cases/case_15")
from kernel import group_quant_dequant

torch.manual_seed(0)
def ref(x, gs=64):
    n_rows, n_cols = x.shape
    out = torch.zeros_like(x)
    for s in range(0, n_cols, gs):
        g = x[:, s:s+gs]
        amax = g.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
        q = torch.round(g / scale).clamp(-127, 127)
        out[:, s:s+gs] = q * scale
    return out

x = torch.randn(4, 100, device="cuda")
y = group_quant_dequant(x, 64)
r = ref(x, 64)
tail = slice(64, 100)
res = {
    "shape": list(y.shape),
    "n_cols": 100, "group_size": 64,
    "tail_allclose": bool(torch.allclose(y[:, tail], r[:, tail])),
    "tail_max_abs_err": float((y[:, tail] - r[:, tail]).abs().max()),
    "tail_kernel_unique_vals": sorted(set(y[:, tail].flatten().tolist())),
    "head_allclose": bool(torch.allclose(y[:, :64], r[:, :64])),
    "head_max_abs_err": float((y[:, :64] - r[:, :64]).abs().max()),
    "allclose_full": bool(torch.allclose(y, r)),
    "max_abs_err_full": float((y - r).abs().max()),
}
import json
print(json.dumps(res))