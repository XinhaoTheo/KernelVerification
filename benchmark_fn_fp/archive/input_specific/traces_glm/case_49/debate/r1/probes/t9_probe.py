
import json, numpy as np, torch
import importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/numerical_cases/case_n/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
coeff_np, points_np = kern.make_inputs_numpy()

ref = np.zeros(8)
for i in range(8):
    x = np.float64(np.float32(points_np[i]))
    s = np.float64(0.0)
    for k in range(48, -1, -1):
        s = s * x + np.float64(coeff_np[i, k])
    ref[i] = s
ref_l2 = float(np.linalg.norm(ref))
floor = 0.001*np.sqrt(8)
denom = max(ref_l2, floor)
print(json.dumps({
  "metric": "reference_L2_norm_vs_floor",
  "ref_l2": ref_l2, "floor": float(floor), "denominator": float(denom),
  "floor_binds": bool(ref_l2 < floor),
  "effective_abs_budget_at_tol": float(0.0002*denom),
  "reference_values": ref.tolist()
}))
