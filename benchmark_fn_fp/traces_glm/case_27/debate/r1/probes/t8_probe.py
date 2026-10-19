import torch, json, sys, os
sys.path.insert(0, "/root/cases/case_28")
from kernel import quant_dequant

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
results = []
# Heavy-tailed rows: bulk ~N(0,1), 2 outlier channels at 50x bulk (in-contract regime)
for ratio in [20, 50, 100]:
    x = torch.randn(4, 1024, device=dev)
    x[:, 0] = ratio
    x[:, 1] = -ratio * 0.9
    y = quant_dequant(x)
    per_row = ((y - x).norm(dim=1) / x.norm(dim=1))
    # identify clamped entries: |y| == 127*scale observed cap
    results.append({
        "outlier_ratio": ratio,
        "per_row_rel_err": [round(v, 4) for v in per_row.tolist()],
        "max_row_err": round(per_row.max().item(), 4),
        "exceeds_5pct": bool((per_row > 0.05).any()),
    })
print(json.dumps({"metric": "per_row_relative_L2_error", "budget": 0.05, "trials": results}))
