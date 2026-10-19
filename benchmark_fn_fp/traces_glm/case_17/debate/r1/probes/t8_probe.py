import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_18/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
x = torch.randn(4, 130, device="cuda", dtype=torch.float32)
out = k.softmax(x)
ref = torch.softmax(x, dim=1)
row_sums = out.sum(dim=1)
print(json.dumps({
  "shape": list(x.shape),
  "row_sums": row_sums.tolist(),
  "max_row_sum_dev_from_1": (row_sums - 1).abs().max().item(),
  "max_abs_err_vs_torch_softmax": (out - ref).abs().max().item(),
  "has_nan": bool(torch.isnan(out).any()),
  "min_entry": out.min().item()
}))