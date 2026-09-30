import json, numpy as np, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_d/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

logits_np, values_np = k.make_inputs_numpy()
logits_t, values_t = k.make_inputs("cuda")
out = k.run(logits_t, values_t).cpu().numpy().astype(np.float64)

l64 = logits_np.astype(np.float64); v64 = values_np.astype(np.float64)
w = np.exp(l64 - l64.max()); p = w / w.sum()
ref = p @ v64

rel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(16))
# probability shift diagnostics
scale = np.abs(l64).max()/7.0
codes = np.clip(np.floor(l64/scale + 0.5), -7, 7)
rounded = codes*scale
wq = np.exp(rounded - rounded.max()); pq = wq/wq.sum()
print(json.dumps({
  "max_abs_logits": float(np.abs(l64).max()),
  "quant_step": float(scale),
  "relative_L2": float(rel),
  "tolerance": 0.02,
  "violation": bool(rel > 0.02),
  "max_prob_shift": float(np.abs(pq-p).max()),
  "ref_norm": float(np.linalg.norm(ref)),
  "output_finite": bool(np.all(np.isfinite(out))),
  "max_abs_out_err": float(np.abs(out-ref).max()),
}))