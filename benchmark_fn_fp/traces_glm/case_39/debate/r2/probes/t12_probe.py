import sys, json, numpy as np
sys.path.insert(0, "/root/numerical_cases/case_d")
import kernel

logits_t, values_t = kernel.make_inputs(device="cuda")
out = kernel.run(logits_t, values_t)
out_np = out.detach().cpu().numpy().astype(np.float64)

l = logits_t.detach().cpu().numpy().astype(np.float64)
v = values_t.detach().cpu().numpy().astype(np.float64)
p = np.exp(l - l.max()); p /= p.sum()
ref = p @ v

rel_l2 = np.linalg.norm(out_np - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(16))
print(json.dumps({
  "metric": "relative L2 error kernel vs fp64 unquantized reference",
  "relative_l2_error": float(rel_l2),
  "tolerance": 0.02,
  "passes_budget": bool(rel_l2 <= 0.02),
  "max_abs_logits": float(np.abs(l).max()),
  "scale": float(np.abs(l).max()/7.0),
  "output_finite": bool(np.all(np.isfinite(out_np))),
  "output_shape": list(out_np.shape),
  "kernel_output": [float(x) for x in out_np],
  "reference": [float(x) for x in ref],
}))