
import json, numpy as np, torch
import importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/numerical_cases/case_n/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

coeff_np, points_np = kern.make_inputs_numpy()
coeff = torch.from_numpy(coeff_np).cuda(); pts = torch.from_numpy(points_np).cuda()
out = kern.run(coeff, pts).cpu().numpy().astype(np.float64)

# float64 reference from stored float32 inputs
ref = np.zeros(8)
for i in range(8):
    x = np.float64(np.float32(points_np[i]))
    s = np.float64(0.0)
    for k in range(48, -1, -1):
        s = s * x + np.float64(coeff_np[i, k])
    ref[i] = s

denom = max(np.linalg.norm(ref), 0.001*np.sqrt(8))
rel = np.linalg.norm(out - ref) / denom
per_abs = np.abs(out - ref)
print(json.dumps({
  "metric": "relative_L2_error_vs_float64_reference",
  "relative_l2": rel, "tolerance": 0.0002, "passes": bool(rel <= 0.0002),
  "reference": ref.tolist(), "kernel_out": out.tolist(),
  "max_abs_err": float(per_abs.max()), "finite": bool(np.all(np.isfinite(out))),
  "ref_l2": float(np.linalg.norm(ref)), "denominator": float(denom)
}))
