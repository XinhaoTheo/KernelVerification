import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_c")
import kernel

logits, values = kernel.make_inputs()
out = kernel.run(logits, values)

l64 = logits.to(torch.float64).numpy()
v64 = values.to(torch.float64).numpy()
p = np.exp(l64 - l64.max()); p /= p.sum()
ref = p @ v64

out_np = out.cpu().numpy().astype(np.float64)
rel = np.linalg.norm(out_np - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(16))
# also simulate the quantization in float64 to isolate fp32 effect
scale = np.abs(l64).max()/7.0
codes = np.clip(np.floor(l64/scale + 0.5), -7, 7)
rounded = codes*scale
w = np.exp(rounded - rounded.max()); q = w/w.sum()
ref_q = q @ v64
rel_q = np.linalg.norm(ref_q - ref) / max(np.linalg.norm(ref), 0.004)

print(json.dumps({
  "metric": "relative L2 error vs unquantized float64 reference (contract)",
  "kernel_rel_l2": float(rel),
  "tolerance": 0.02,
  "quantized_fp64_rel_l2": float(rel_q),
  "finite": bool(np.all(np.isfinite(out_np))),
  "shape": list(out_np.shape),
  "max_abs_logit": float(np.abs(l64).max()),
  "passes_contract": bool(rel <= 0.02 and np.all(np.isfinite(out_np)))
}))