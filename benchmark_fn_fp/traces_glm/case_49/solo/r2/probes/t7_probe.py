
import json, torch, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_n/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

coeffs, pts = k.make_inputs()
out = k.run(coeffs, pts)

# float64 reference from stored float32 inputs
c64 = coeffs.cpu().numpy().astype(np.float64)
p64 = pts.cpu().numpy().astype(np.float64)
ref = np.array([sum(c64[i,j] * p64[i]**j for j in range(49)) for i in range(8)])

o = out.cpu().numpy().astype(np.float64)
err = np.linalg.norm(o - ref)
den = max(np.linalg.norm(ref), 0.001*np.sqrt(8))
rel = err / den
finite = bool(np.all(np.isfinite(o)))

print(json.dumps({
  "metric": "relative L2 error vs float64 reference on fixed make_inputs workload",
  "rel_l2": rel, "threshold": 0.0002, "finite": finite,
  "ref_norm": float(np.linalg.norm(ref)), "err_norm": float(err),
  "ref": ref.tolist(), "out": o.tolist(), "shape": list(out.shape), "dtype": str(out.dtype)
}))
