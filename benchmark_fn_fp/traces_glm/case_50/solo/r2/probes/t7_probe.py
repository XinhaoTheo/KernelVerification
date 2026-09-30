import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_o")
import kernel

u, b = kernel.make_inputs(device="cuda")
out = kernel.run(u, b)
torch.cuda.synchronize()

u64 = u.cpu().numpy().astype(np.float64)
b64 = b.cpu().numpy().astype(np.float64)
alpha = (u64*b64).sum() / (u64*u64).sum()
res = b64 - alpha*u64
ref = res / np.linalg.norm(res)
o = out.cpu().numpy()
err = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
  "metric": "relative L2 error vs float64 reference (contract bound 0.01)",
  "shape": list(o.shape),
  "finite": bool(np.isfinite(o).all()),
  "rel_l2_error": float(err),
  "pass": bool(err <= 0.01),
  "max_abs_err": float(np.abs(o-ref).max()),
  "residual_norm": float(np.linalg.norm(res)),
  "dtype": str(out.dtype)
}))
