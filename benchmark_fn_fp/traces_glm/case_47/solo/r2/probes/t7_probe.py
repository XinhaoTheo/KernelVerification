
import sys, json
sys.path.insert(0, "/root/numerical_cases/case_l")
import numpy as np, torch
import kernel

matrix, rhs = kernel.make_inputs()
out = kernel.run(matrix, rhs)
# float64 exact solution from original float32 values
M64 = matrix.detach().cpu().numpy().astype(np.float64)
b64 = rhs.detach().cpu().numpy().astype(np.float64)
x = np.linalg.solve(M64, b64)
o = out.detach().cpu().numpy().astype(np.float64)
rel = np.linalg.norm(o - x) / max(np.linalg.norm(x), 0.001*np.sqrt(16))
# theory check: Richardson factor per eigenvalue
factors = [(1-l)**64 for l in [0.03125,0.0625,0.125,0.25]]
print(json.dumps({
  "relative_L2_error": float(rel),
  "tolerance": 0.08,
  "exceeds": bool(rel > 0.08),
  "norm_x": float(np.linalg.norm(x)),
  "finite": bool(np.all(np.isfinite(o))),
  "worst_richardson_factors": factors,
}))
