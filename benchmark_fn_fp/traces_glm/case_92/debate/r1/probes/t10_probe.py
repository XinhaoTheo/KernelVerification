import torch, numpy as np, json

# all finite BF16 values in [-10,10]
b = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float().numpy()
mask = np.isfinite(b) & (np.abs(b) <= 10)
vals = np.unique(b[mask])
x64 = torch.tensor(vals.tolist(), dtype=torch.float64)

integ = torch.floor(x64 + 0.5)
frac = x64 - integ

c = [0.00133909, 0.00967603, 0.0555036, 0.240221, 0.693147, 1.0]
cd = [torch.tensor(v, dtype=torch.float64) for v in c]
p = cd[0].expand_as(x64).clone()
for k in range(1, 6):
    p = p * frac + cd[k]

target = torch.exp2(x64)
rel = (p - target).abs() / target
maxidx = int(torch.argmax(rel))
near = (frac.abs() > 0.45).nonzero().squeeze(-1)
res = {
    "metric": "pure FP64 polynomial relative error vs exp2 (truncation error only), exhaustive BF16 domain",
    "n_bf16_values": len(vals),
    "max_rel_err": float(rel.max()),
    "max_err_at_x": float(vals[maxidx]),
    "count_over_1e-6": int((rel > 1e-6).sum()),
    "max_rel_err_frac_abs_gt_0.45": float(rel[near].max()) if near.numel() else None,
    "tolerance": 1e-6,
}
print(json.dumps(res))
