
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_27/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
R, C = 4, 8
x = torch.randn(R, C, device="cuda", dtype=torch.float32)
mask = torch.ones(R, C, device="cuda", dtype=torch.bool)
mask[1] = False              # fully masked row
mask[2, 3:] = False          # partially masked row
y = m.masked_softmax(x, mask)

# reference
e = torch.where(mask, torch.exp(x - torch.where(mask, x, torch.full_like(x, -float('inf'))).max(dim=1, keepdim=True).values), torch.zeros_like(x))
s = e.sum(dim=1, keepdim=True)
ref = torch.where(s > 0, e / s.clamp_min(1e-30), torch.zeros_like(e))
ref = torch.where(mask, ref, torch.zeros_like(ref))

out = {
 "y_row_fullmasked": y[1].tolist(),
 "nan_count": int(torch.isnan(y).sum()),
 "inf_count": int(torch.isinf(y).sum()),
 "all_finite": bool(torch.isfinite(y).all()),
 "fullmasked_all_zero": bool((y[1] == 0).all()),
 "max_abs_err_other_rows": float((y[[0,2,3]] - ref[[0,2,3]]).abs().max()),
 "row_sums": y.sum(dim=1).tolist(),
}
print(json.dumps(out, indent=1))
