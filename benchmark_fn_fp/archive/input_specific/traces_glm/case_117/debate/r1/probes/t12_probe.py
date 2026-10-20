import numpy as np, json
import torch
import sys
sys.path.insert(0, '/root/numerical_cases/case_m')
import kernel as k

coeffs, points = k.make_inputs(device='cuda')
out = k.run(coeffs, points)
out_np = out.detach().cpu().numpy().astype(np.float64)

# fp64 reference from stored fp32 inputs (stored c[i,0])
c64 = coeffs.detach().cpu().numpy().astype(np.float64)
p64 = points.detach().cpu().numpy().astype(np.float64)
ref = np.array([sum(c64[i,j] * p64[i]**j for j in range(49)) for i in range(8)])

denom = max(np.linalg.norm(ref), 0.001*np.sqrt(8))
rel = np.linalg.norm(out_np - ref) / denom
print(json.dumps({
  "kernel_output": out_np.tolist(),
  "reference_fp64": ref.tolist(),
  "abs_err_per_row": (out_np - ref).tolist(),
  "norm_ref": float(np.linalg.norm(ref)),
  "floor": 0.001*np.sqrt(8),
  "denominator": float(denom),
  "rel_l2_error": float(rel),
  "tolerance": 0.0002,
  "passes": bool(rel <= 0.0002),
  "finite": bool(np.all(np.isfinite(out_np))),
}))