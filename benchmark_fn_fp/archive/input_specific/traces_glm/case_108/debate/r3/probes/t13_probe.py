import json, numpy as np, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_d/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

logits_np, values_np = k.make_inputs_numpy()
l32 = logits_np.astype(np.float32); v32 = values_np.astype(np.float32)
# fp32 pipeline: unquantized softmax + fp32 accumulation (numpy fp32 end-to-end)
w32 = np.exp(l32 - l32.max(), dtype=np.float32); p32 = w32 / w32.sum(dtype=np.float32)
out32 = (p32[:,None] * v32).sum(axis=0, dtype=np.float32).astype(np.float64)

l64 = logits_np.astype(np.float64); v64 = values_np.astype(np.float64)
w = np.exp(l64 - l64.max()); p = w / w.sum()
ref = p @ v64

rel = np.linalg.norm(out32 - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(16))
print(json.dumps({
  "fp32_only_relative_L2": float(rel),
  "tolerance": 0.02,
  "negligible": bool(rel < 1e-4),
  "max_abs_err": float(np.abs(out32-ref).max()),
  "ref_norm": float(np.linalg.norm(ref)),
}))