import torch, sys, json
sys.path.insert(0, "/root/cases/case_15")
from kernel import group_quant_dequant

def ref_half_up(x, gs):
    n_rows, n_cols = x.shape
    out = torch.zeros_like(x)
    for s in range(0, n_cols, gs):
        g = x[:, s:s+gs]
        amax = g.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
        q = torch.floor(g / scale + 0.5).clamp(-127, 127)
        out[:, s:s+gs] = q * scale
    return out

def ref_half_even(x, gs):
    n_rows, n_cols = x.shape
    out = torch.zeros_like(x)
    for s in range(0, n_cols, gs):
        g = x[:, s:s+gs]
        amax = g.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
        q = torch.round(g / scale).clamp(-127, 127)  # half-to-even
        out[:, s:s+gs] = q * scale
    return out

# Construct exact tie points: scale must be such that x/scale = k+0.5 exactly.
# Choose group of 64: put values so absmax = 127 (so scale=1), and include values at exact .5 offsets.
n_rows, n_cols = 2, 64
x = torch.zeros(n_rows, n_cols, device="cuda")
tievals = [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, -3.5]
for i, v in enumerate(tievals):
    x[0, i] = v
x[1] = torch.tensor([127.0] + [0.5]*(n_cols-1), device="cuda", dtype=torch.float32)

y = group_quant_dequant(x, 64)
hu = ref_half_up(x, 64)
he = ref_half_even(x, 64)
res = {
    "tie_count": len(tievals),
    "kernel_vs_halfup_max_err": float((y - hu).abs().max()),
    "kernel_vs_halfeven_allclose": bool(torch.allclose(y, he)),
    "kernel_vs_halfeven_max_err": float((y - he).abs().max()),
    "kernel_vs_halfeven_mismatch_count": int((~torch.isclose(y, he)).sum()),
    "sample_x": x[0, :9].tolist(),
    "sample_kernel": y[0, :9].tolist(),
    "sample_halfeven": he[0, :9].tolist(),
}
print(json.dumps(res))