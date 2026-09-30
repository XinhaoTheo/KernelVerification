import importlib.util, json, math
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_i/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

(x,) = k.make_inputs("cuda")
out = k.run(x)
torch.cuda.synchronize()

xd = x.cpu().numpy().astype(np.float64)
mean64 = xd.sum() / 128
var64 = ((xd - mean64)**2).sum() / 128
ref = (xd - mean64) / math.sqrt(var64 + 1e-5)
den64 = math.sqrt(var64 + 1e-5)

od = out.cpu().numpy().astype(np.float64)
rel_l2 = np.linalg.norm(od - ref) / max(np.linalg.norm(ref), 0.001*math.sqrt(128))

# reconstruct kernel's internal one-pass variance in float32 (same sequential order)
tot = np.float32(0.0); sq = np.float32(0.0)
for v in x.cpu().numpy():
    vf = np.float32(v)
    tot = np.float32(tot + vf)
    sq = np.float32(sq + np.float32(vf*vf))
mean_f = np.float32(tot / 128)
var_f = np.float32(max(np.float32(sq/128 - mean_f*mean_f), 0.0))
den_f = np.float32(np.sqrt(np.float32(var_f + 1e-5)))

# effective denominator implied by actual kernel output: ratio ||out||/||ref_numerator||
eff_den = np.linalg.norm(od) / np.linalg.norm((xd - mean64))  # out = num/den => den = ||num||/||out||

print(json.dumps({
  "metric": "rel_L2 + variance/denominator decomposition vs float64 reference",
  "input": "make_inputs seed 782515, shape (128,), float32",
  "rel_l2": float(rel_l2), "tolerance": 0.02, "within_tol": bool(rel_l2 <= 0.02),
  "finite": bool(np.isfinite(od).all()), "shape": list(od.shape),
  "var64": float(var64), "var_f32_onepass": float(var_f),
  "var_rel_err": float(abs(var_f - var64)/var64),
  "den64": float(den64), "den_f32": float(den_f), "eff_den_from_output": float(eff_den),
  "den_rel_err": float(abs(eff_den - den64)/den64),
  "exceeds_0.02": bool(rel_l2 > 0.02)
}))