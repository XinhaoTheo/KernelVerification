import sys, json
sys.path.insert(0, "/root/evidence_cases/case_e20")
from fractions import Fraction
import numpy as np, torch
import kernel

x, = kernel.make_inputs()
before = x.clone()
out = kernel.run(x)
out_np = out.detach().cpu().numpy().astype(np.float64)
xn = x.detach().cpu().numpy()
target = []
for row in xn:
    s = Fraction(0)
    for v in row:
        s += Fraction(np.float64(v))
    target.append(float(s))
t64 = np.array(target)
err = float(np.linalg.norm(out_np - t64) / max(np.linalg.norm(t64), 1e-12))
print(json.dumps({
  "metric": "relative L2 error vs exact real-number target (Fraction)",
  "target": target,
  "output": out_np.tolist(),
  "relative_error": err,
  "tolerance": 1e-5,
  "inputs_unmodified": bool(torch.equal(x, before)),
  "finite": bool(np.isfinite(out_np).all()),
  "passed": bool(err <= 1e-5 and torch.equal(x, before) and np.isfinite(out_np).all())
}))