import sys, json
sys.path.insert(0, "/root/numerical_cases/case_e")
import numpy as np, torch
from kernel import run, make_inputs_numpy

matrix, initial, drive = make_inputs_numpy()
m64 = matrix.astype(np.float64); i64 = initial.astype(np.float64); d64 = drive.astype(np.float64)
ref = i64.copy()
for t in range(64):
    ref = m64 @ ref + d64[t]

tm = torch.from_numpy(matrix).cuda(); ti = torch.from_numpy(initial).cuda(); td = torch.from_numpy(drive).cuda()
out = run(tm, ti, td).cpu().numpy().astype(np.float64)

err = np.linalg.norm(out - ref)
denom = max(np.linalg.norm(ref), 0.001*np.sqrt(16))
rel = err / denom
print(json.dumps({
  "metric": "relative L2 error vs float64 reference (contract metric)",
  "rel_l2": rel, "abs_l2": err, "ref_norm": np.linalg.norm(ref),
  "tolerance": 0.002, "passes": bool(rel <= 0.002),
  "finite": bool(np.all(np.isfinite(out))), "shape": list(out.shape),
  "max_abs_elem_err": float(np.max(np.abs(out - ref))),
}))