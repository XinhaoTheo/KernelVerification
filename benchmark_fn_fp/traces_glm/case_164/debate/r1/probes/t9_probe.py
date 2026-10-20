import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_164")

# exhaustive sweep of all finite BF16 values in [-10,10] through the actual kernel
vals = []
for bits in range(65536):
    b = torch.tensor([bits], dtype=torch.int16).view(torch.bfloat16).item()
    if np.isfinite(b) and abs(b) <= 10:
        vals.append(b)
vals = sorted(set(vals))
x = torch.tensor(vals, dtype=torch.bfloat16, device="cuda")
out = run(x)
target = torch.exp2(x.to(torch.float64)).cpu()
act = out.cpu().to(torch.float64)
rel = (act - target).abs() / target
over = (rel > 1e-6)
maxidx = int(torch.argmax(rel))
vals_sorted = vals
res = {
    "metric": "kernel FP32 output relative error vs FP64 exp2, exhaustive BF16 domain sweep",
    "n_values": len(vals),
    "max_rel_err": float(rel.max()),
    "max_err_at_x": vals_sorted[maxidx],
    "count_over_1e-6": int(over.sum()),
    "tolerance": 1e-6,
}
print(json.dumps(res))
