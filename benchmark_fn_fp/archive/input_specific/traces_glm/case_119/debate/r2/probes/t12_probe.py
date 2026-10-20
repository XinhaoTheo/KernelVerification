import sys, json
sys.path.insert(0, "/root/numerical_cases/case_o")
import numpy as np, torch
import kernel as K

u, b = K.make_inputs()
out = K.run(u, b).cpu().numpy()

u64 = u.cpu().numpy().astype(np.float64)
b64 = b.cpu().numpy().astype(np.float64)
alpha = (u64*b64).sum() / (u64*u64).sum()
res = b64 - alpha*u64
ref = res / np.linalg.norm(res)

rel = np.linalg.norm(out.astype(np.float64) - ref) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
  "metric": "relative L2 error vs float64 reference on fixed workload",
  "rel_l2_error": float(rel),
  "tolerance": 0.01,
  "alpha_f64": float(alpha),
  "residual_norm": float(np.linalg.norm(res)),
  "finite_output": bool(np.isfinite(out).all()),
  "shape": list(out.shape),
  "max_abs_err": float(np.abs(out - ref).max()),
}))