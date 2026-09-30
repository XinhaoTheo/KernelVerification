import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_27/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x = torch.randn(3, 8, device="cuda", dtype=torch.float32)
mask = torch.ones(3, 8, dtype=torch.bool, device="cuda")
mask[1] = False  # fully masked row
y = k.masked_softmax(x, mask)
torch.cuda.synchronize()
row1 = y[1]
res = {
  "full_row_isnan": bool(torch.isnan(row1).all().item()),
  "full_row_values": row1.tolist(),
  "nan_count_total": int(torch.isnan(y).sum().item()),
  "finite_elsewhere": bool(torch.isfinite(y[[0,2]].all().item()),
}
print(json.dumps(res))
