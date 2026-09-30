import torch, json, sys
sys.path.insert(0, "/root/cases/case_28")
from kernel import quant_dequant

torch.manual_seed(0)
rows, cols = 64, 512
x = torch.randn(rows, cols, device="cuda") * 0.5
# heavy-tailed: 3 outlier channels per row, 1-2 orders of magnitude above bulk
for r in range(rows):
    idx = torch.randperm(cols, device="cuda")[:3]
    x[r, idx] *= torch.tensor([30.0, 60.0, 100.0], device="cuda")

y = quant_dequant(x)
err = (y - x).norm(dim=1) / x.norm(dim=1)
print(json.dumps({
    "metric": "per-row relative reconstruction error",
    "max_rel_err": err.max().item(),
    "mean_rel_err": err.mean().item(),
    "frac_rows_over_5pct": (err > 0.05).float().mean().item(),
    "min_rel_err": err.min().item(),
}))