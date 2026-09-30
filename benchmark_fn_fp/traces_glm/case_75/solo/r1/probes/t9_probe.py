import json, math, sys
sys.path.insert(0, "/root/evidence_cases/case_e14")
import kernel
x, = kernel.make_inputs()
before = x.clone()
out = kernel.run(x)
xd = before.cpu().numpy().astype(np.float64) if False else None
import numpy as np
vals = before.detach().cpu().numpy().astype(np.float64)
target = np.array([math.fsum(row) for row in vals])
actual = out.detach().cpu().numpy().astype(np.float64)
err = float(np.linalg.norm(actual - target) / max(np.linalg.norm(target), 1e-12))
print(json.dumps({
  "rows": vals.tolist(),
  "target": target.tolist(),
  "output": actual.tolist(),
  "relative_error": err,
  "tolerance": 1e-5,
  "unmodified": bool(torch.equal(x, before)) if False else bool((x == before).all()),
  "shape": list(out.shape), "dtype": str(out.dtype), "finite": bool(np.isfinite(actual).all()),
  "passes": bool(err <= 1e-5 and np.isfinite(actual).all() and tuple(out.shape)==(4,) and out.dtype==__import__('torch').float32 and (x==before).all())
}))
