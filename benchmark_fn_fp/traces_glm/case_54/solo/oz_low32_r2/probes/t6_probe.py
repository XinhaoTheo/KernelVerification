
import json, sys
sys.path.insert(0, "/root/numerical_cases/case_s")
import numpy as np, torch
from kernel import make_inputs, run

a, f, p = make_inputs()
out = run(a, f, p).cpu().numpy().astype(np.float64)
A = a.cpu().numpy().astype(np.float64)
F = f.cpu().numpy().astype(np.float64)
P = p.cpu().numpy().astype(np.float64)
ref = 1.0 + ((A * (np.cos(P) - np.cos(P + F)) / F).sum(axis=1))
rel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-12)
# also per-row errors
rowerr = np.abs(out - ref)
print(json.dumps({
  "metric": "relative L2 error vs float64 analytic integral",
  "output": out.tolist(), "reference": ref.tolist(),
  "per_row_abs_err": rowerr.tolist(),
  "rel_l2": float(rel), "tolerance": 0.035,
  "max_frequency": float(F.max()), "passes": bool(rel <= 0.035),
  "finite": bool(np.all(np.isfinite(out))), "shape": list(out.shape)
}))
