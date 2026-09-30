
import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_c")
import kernel

logits, values = kernel.make_inputs("cuda")
out = kernel.run(logits, values)

l64 = logits.double().cpu().numpy()
v64 = values.double().cpu().numpy()
w = np.exp(l64 - l64.max())
p = w / w.sum()
ref = p @ v64

out_np = out.double().cpu().numpy()
denom = max(np.linalg.norm(ref), 0.001*np.sqrt(16))
rel = np.linalg.norm(out_np - ref) / denom
print(json.dumps({
  "metric": "relative L2 error vs float64 unquantized reference (contract metric)",
  "rel_l2": float(rel),
  "threshold": 0.02,
  "passes": bool(rel <= 0.02),
  "output_shape": list(out.shape),
  "finite": bool(np.isfinite(out_np).all()),
  "ref_norm": float(np.linalg.norm(ref)),
  "max_abs_err": float(np.abs(out_np-ref).max()),
}))
