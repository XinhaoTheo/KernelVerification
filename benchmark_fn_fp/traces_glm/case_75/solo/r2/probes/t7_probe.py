import json, math, sys
sys.path.insert(0, "/root/evidence_cases/case_e14")
import kernel

x, = kernel.make_inputs("cuda")
before = x.clone()
out = kernel.run(x)
actual = out.detach().cpu().numpy().astype(np.float64) if False else out.cpu().numpy().astype(np.float64)
vals = before.cpu().numpy().astype(np.float64)
# exact real-number sum of stored values
target = [math.fsum(row) for row in vals]
# also record the stored small-value content per row for context
import numpy as np
num = np.linalg.norm
err = float(num(np.array(actual) - np.array(target)) / max(num(np.array(target)), 1e-12))
structural = (tuple(out.shape)==(4,) and out.dtype==torch.float32 and bool(np.isfinite(actual).all()) and bool((x==before).all()))
print(json.dumps({
  "metric": "l2 relative error vs exact fsum target",
  "target_exact": target,
  "output": actual.tolist(),
  "relative_error": err,
  "tolerance": 1e-5,
  "row_contents_row0": vals[0].tolist(),
  "shape_dtype_finite_unmodified": structural,
  "passes_contract": bool(structural and err <= 1e-5)
}))