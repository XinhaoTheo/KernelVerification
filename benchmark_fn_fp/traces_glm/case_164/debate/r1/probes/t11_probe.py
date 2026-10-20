import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_164")
from kernel import run

b = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float().numpy()
mask = np.isfinite(b) & (np.abs(b) <= 10)
vals = np.unique(b[mask])

x = torch.tensor(vals.tolist(), dtype=torch.bfloat16, device="cuda")
out = run(x)
target = torch.exp2(x.to(torch.float64)).cpu()
act = out.cpu().to(torch.float64)
rel = (act - target).abs() / target
maxidx = int(torch.argmax(rel))
res = {
    "metric": "actual kernel FP32 output relative error vs FP64 exp2, exhaustive BF16 domain sweep",
    "n_values": int(len(vals)),
    "max_rel_err": float(rel.max()),
    "max_err_at_x": float(vals[maxidx]),
    "count_over_1e-6": int((rel > 1e-6).sum()),
    "tolerance": 1e-6,
    "argmax_x_bits": int(x[maxidx].view(torch.int16).item() & 0xFFFF),
}
print(json.dumps(res))
