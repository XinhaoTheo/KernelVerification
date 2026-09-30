import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_18/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
out = k.softmax(x)
ref = torch.softmax(x, dim=1)
row_sums = out.sum(dim=1)
print(json.dumps({
  "shape": list(x.shape),
  "row_sums": row_sums.tolist(),
  "has_nan": bool(torch.isnan(out).any()),
  "has_inf": bool(torch.isinf(out).any()),
  "max_abs_err_vs_torch_softmax": (out - ref).abs().max().item()
}))