
import json, sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_18/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
res = {}
for n_cols in [130, 200, 257, 256, 128]:
    x = torch.randn(4, n_cols, device="cuda", dtype=torch.float32)
    y = k.softmax(x, 128)
    ref = torch.softmax(x, dim=-1)
    rows = y.sum(-1)
    res[str(n_cols)] = {
        "row_sums": [float(v) for v in rows],
        "max_row_sum_dev_from_1": float((rows-1).abs().max()),
        "max_abs_err_vs_torch": float((y-ref).abs().max()),
        "max_rel_ratio_vs_torch": float((y/ref).max()),
        "nonfinite": int((~torch.isfinite(y)).sum()),
        "allclose_1e-5": bool(torch.allclose(y, ref, atol=1e-5, rtol=1e-5)),
        "predicted_ratio_total_over_partial": float((torch.exp(x-x.max(-1,keepdim=True).values).sum(-1) /
             torch.exp(x[:, :(n_cols//128)*128]-x.max(-1,keepdim=True).values).sum(-1)).max()) if n_cols>=128 else None,
    }
print(json.dumps(res))
