import sys, json
sys.path.insert(0, "/root/numerical_cases/case_v")
import numpy as np, torch
import kernel as K

coefficients, offset = K.make_inputs("cuda")
out = K.run(coefficients, offset).cpu().numpy().astype(np.float64)
c = coefficients.cpu().numpy().astype(np.float64)
o = float(offset.cpu().numpy()[0])
n = np.arange(64)
ref = np.full(64, o)
for k in range(1, 17):
    ref += c[0, k-1]*np.cos(2*np.pi*k*n/64) + c[1, k-1]*np.sin(2*np.pi*k*n/64)
diff = out - ref
ratio = np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
  "metric": "relative L2 error vs float64 all-16-mode reference",
  "ratio": float(ratio),
  "tolerance": 0.15,
  "exceeds": bool(ratio > 0.15),
  "ref_norm": float(np.linalg.norm(ref)),
  "out_shape": list(out.shape),
  "out_finite": bool(np.isfinite(out).all()),
  "is_contiguous": bool(coefficients.is_contiguous()),
}))