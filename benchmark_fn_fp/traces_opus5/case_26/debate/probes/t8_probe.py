
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_27/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev = "cuda"
n_rows, n_cols = 4, 8
x = torch.randn(n_rows, n_cols, device=dev, dtype=torch.float32)
mask = torch.rand(n_rows, n_cols, device=dev) > 0.5
mask[0] = False           # fully padded row (in-domain per problem.txt)
mask[1] = True
mask[2, 3] = True         # ensure >=1 kept
mask[3, 0] = True

y = m.masked_softmax(x, mask)
torch.cuda.synchronize()

row0 = y[0]
kept_counts = mask.sum(1).tolist()

# spec reference: zero rows where nothing kept
e = torch.where(mask, torch.exp(x - torch.where(mask, x, torch.full_like(x, -float("inf"))).max(1, keepdim=True).values), torch.zeros_like(x))
den = e.sum(1, keepdim=True)
ref = torch.where(den > 0, e / den.clamp_min(1e-45), torch.zeros_like(e))

out = {
  "metric": "finiteness + exact-zero on empty-mask row (contract: all-zero, all-finite)",
  "kept_counts_per_row": kept_counts,
  "empty_row_values": [float(v) for v in row0.tolist()],
  "empty_row_num_nan": int(torch.isnan(row0).sum()),
  "empty_row_num_inf": int(torch.isinf(row0).sum()),
  "empty_row_all_zero": bool(torch.all(row0 == 0)),
  "empty_row_all_finite": bool(torch.isfinite(row0).all()),
  "whole_tensor_num_nonfinite": int((~torch.isfinite(y)).sum()),
  "nonempty_rows_max_abs_err": float((y[1:] - ref[1:]).abs().max()),
  "shape": list(y.shape), "dtype": str(y.dtype),
}
print(json.dumps(out))
