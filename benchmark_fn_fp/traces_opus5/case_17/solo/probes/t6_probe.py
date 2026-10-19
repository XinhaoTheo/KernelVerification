
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_18/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res = {}
torch.manual_seed(0)
for n_cols in [256, 300, 129, 100]:
    x = torch.randn(4, n_cols, device='cuda', dtype=torch.float32)
    y = m.softmax(x, 128)
    ref = torch.softmax(x, dim=1)
    res[str(n_cols)] = dict(
        row_sums=y.sum(1).tolist(),
        max_abs_err=float((y-ref).abs().max()),
        max_rel_err=float(((y-ref).abs()/ref.abs().clamp_min(1e-30)).max()),
    )
print(json.dumps({"metric":"row sum == 1 and elementwise vs torch.softmax","results":res}, indent=1))
