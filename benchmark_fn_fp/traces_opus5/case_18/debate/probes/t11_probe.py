
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_18/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
x = (torch.randn(4, 200, device="cuda", dtype=torch.float32) * 0.1)
x[:, -3:] = 20.0
y = k.softmax(x, 128)
ref = torch.softmax(x, dim=-1)
res = {
    "row_sums": [float(v) for v in y.sum(-1)],
    "max_out_value": float(y.max()),
    "ref_max_out_value": float(ref.max()),
    "max_abs_err": float((y-ref).abs().max()),
    "max_ratio_vs_torch": float((y/ref.clamp_min(1e-45)).max()),
    "nonfinite": int((~torch.isfinite(y)).sum()),
    "entries_gt_1": int((y > 1.0).sum()),
    "shape": list(y.shape),
}
print(json.dumps(res))
