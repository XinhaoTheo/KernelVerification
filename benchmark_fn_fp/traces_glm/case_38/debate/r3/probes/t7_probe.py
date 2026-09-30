import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_c")
import kernel

logits_t, values_t = kernel.make_inputs("cuda")
out = kernel.run(logits_t, values_t).cpu().numpy().astype(np.float64)

l = logits_t.cpu().numpy().astype(np.float64)
v = values_t.cpu().numpy().astype(np.float64)
w = np.exp(l - l.max()); p = w / w.sum()
ref = p @ v

denom = max(np.linalg.norm(ref), 0.001*np.sqrt(16))
rel = np.linalg.norm(out - ref) / denom
print(json.dumps({
  "metric": "relative_L2_error",
  "reason": "contract in problem.txt: error = ||out-ref||/max(||ref||,0.001*4) must be <= 0.02",
  "relative_L2_error": rel,
  "tolerance": 0.02,
  "bound_exceeded": bool(rel > 0.02),
  "max_abs_err": float(np.max(np.abs(out - ref))),
  "quantization_step": float(np.max(np.abs(l))/7.0),
  "max_abs_logit": float(np.max(np.abs(l))),
  "ref_norm": float(np.linalg.norm(ref)),
  "output_finite": bool(np.all(np.isfinite(out))),
  "output_shape": list(out.shape),
}))