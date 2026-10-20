
import sys, json
sys.path.insert(0, "/root/numerical_cases/case_m")
import numpy as np, torch
from kernel import make_inputs, run

coeffs, points = make_inputs()
out = run(coeffs, points)
torch.cuda.synchronize()
out = out.detach().cpu().numpy().astype(np.float64)
c = coeffs.cpu().numpy().astype(np.float64)
p = points.cpu().numpy().astype(np.float64)
ref = np.array([np.sum(c[i] * p[i]**np.arange(49)) for i in range(8)])
denom = max(np.linalg.norm(ref), 0.001*np.sqrt(8))
rel = np.linalg.norm(out - ref) / denom
print(json.dumps({
  "metric": "relative L2 error vs float64 reference, contract denominator",
  "output": out.tolist(), "reference": ref.tolist(),
  "ref_norm": float(np.linalg.norm(ref)), "denominator": float(denom),
  "abs_err": float(np.linalg.norm(out-ref)), "relative_error": float(rel),
  "tolerance": 0.0002, "passes": bool(rel <= 0.0002),
  "finite": bool(np.all(np.isfinite(out))), "shape": list(out.shape)
}))
