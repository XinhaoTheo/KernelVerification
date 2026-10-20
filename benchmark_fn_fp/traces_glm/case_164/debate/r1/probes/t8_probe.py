import torch, numpy as np, json

# all finite BF16 values in [-10,10]
vals = []
for bits in range(65536):
    b = torch.tensor([bits], dtype=torch.int16).view(torch.bfloat16).item()
    if np.isfinite(b) and abs(b) <= 10:
        vals.append(b)
vals = sorted(set(vals))
x64 = torch.tensor(vals, dtype=torch.float64)

# exact stored value -> integer = floor(x+0.5), fraction = x - integer
integ = torch.floor(x64 + 0.5)
frac = x64 - integ

# pure polynomial in FP64 (coefficients as exact as given, decimal strings)
c = [0.00133909, 0.00967603, 0.0555036, 0.240221, 0.693147, 1.0]
cd = [torch.tensor(v, dtype=torch.float64) for v in c]
p = torch.full_like(x64, 1.0) * cd[0]
# Horner: p = 0.00133909 then successive *frac + next
p = cd[0].expand_as(x64).clone()
for k in range(1, 6):
    p = p * frac + cd[k]

target = torch.exp2(x64)
rel = (p - target).abs() / target
maxidx = int(torch.argmax(rel))
over = int((rel > 1e-6).sum())
near = (frac.abs() > 0.49).nonzero().squeeze(-1)
rel_near = rel[near] if near.numel() else torch.tensor([0.0])
res = {
    "metric": "pure FP64 polynomial relative error vs exp2 (truncation error only)",
    "n_bf16_values": len(vals),
    "max_rel_err": float(rel.max()),
    "max_err_at_x": vals[maxidx],
    "count_over_1e-6": over,
    "max_rel_err_frac_gt_0.49": float(rel_near.max()) if near.numel() else None,
    "tolerance": 1e-6,
}
print(json.dumps(res))
