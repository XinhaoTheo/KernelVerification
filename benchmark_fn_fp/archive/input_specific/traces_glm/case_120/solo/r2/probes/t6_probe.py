import json, sys, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_p")
import kernel

u, b = kernel.make_inputs("cuda")
out = kernel.run(u, b)
torch.cuda.synchronize()

u64 = u.cpu().numpy().astype(np.float64)
b64 = b.cpu().numpy().astype(np.float64)
alpha = (u64*b64).sum() / (u64*u64).sum()
res = b64 - alpha*u64
ref = res / np.linalg.norm(res)

o = out.cpu().numpy().astype(np.float64)
err = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
  "shape": list(out.shape),
  "finite": bool(np.isfinite(o).all()),
  "relative_l2": float(err),
  "tolerance": 0.01,
  "passes": bool(err <= 0.01 and out.shape == (32,) and np.isfinite(o).all()),
  "alpha": float(alpha),
  "resid_norm": float(np.linalg.norm(res))
}))
