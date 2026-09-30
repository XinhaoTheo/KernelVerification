import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e20/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, = k.make_inputs()
before = x.clone()
out = k.run(x)
# exact real-number target: convert stored fp32 values to float64 (exact) and sum in high precision
vals = before.detach().cpu().numpy().astype(np.float64)
target = vals.sum(axis=1)
actual = out.detach().cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(actual - target) / max(np.linalg.norm(target), 1e-12))
print(json.dumps({
  "output": actual.tolist(),
  "exact_target": target.tolist(),
  "relative_error": rel,
  "tolerance": 1e-5,
  "input_unmodified": bool(torch.equal(x, before)),
  "shape": list(out.shape), "dtype": str(out.dtype),
  "finite": bool(np.isfinite(actual).all()),
  "passes_contract": bool(rel <= 1e-5)
}))