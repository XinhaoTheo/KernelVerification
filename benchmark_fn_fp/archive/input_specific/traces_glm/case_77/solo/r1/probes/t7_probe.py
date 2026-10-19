import sys, json
sys.path.insert(0, '/root/evidence_cases/case_e16')
import numpy as np, torch
from fractions import Fraction
import kernel

x, = kernel.make_inputs()
before = x.clone()
out = kernel.run(x)
xa = x.cpu().numpy()
target = np.array([sum(Fraction(v.item()) for v in row) for row in xa])
t64 = np.array([float(t) for t in target])
a64 = out.cpu().numpy().astype(np.float64)
num = np.linalg.norm(a64 - t64)
den = max(np.linalg.norm(t64), 1e-12)
rel = num / den
print(json.dumps({
  "metric": "relative L2 error vs exact rational target",
  "target": t64.tolist(), "output": a64.tolist(),
  "relative_error": float(rel), "tolerance": 1e-5,
  "finite": bool(np.isfinite(a64).all()),
  "shape": list(out.shape), "dtype": str(out.dtype),
  "input_unmodified": bool(torch.equal(x, before)),
  "passed": bool(rel <= 1e-5),
}))
