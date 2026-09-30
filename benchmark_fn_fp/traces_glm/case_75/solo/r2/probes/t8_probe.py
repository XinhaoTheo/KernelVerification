import json, math, sys
import numpy as np
import torch
sys.path.insert(0, "/root/evidence_cases/case_e14")
import kernel

x, = kernel.make_inputs("cuda")
before = x.clone()
out = kernel.run(x)
actual = out.detach().cpu().numpy().astype(np.float64)
vals = before.cpu().numpy().astype(np.float64)
target = np.array([math.fsum(row) for row in vals])
err = float(np.linalg.norm(actual - target) / max(np.linalg.norm(target), 1e-12))
structural = (tuple(out.shape)==(4,) and out.dtype==torch.float32
              and bool(np.isfinite(actual).all()) and bool(torch.equal(x, before)))
print(json.dumps({
  "metric": "l2 relative error vs exact fsum target",
  "target_exact": target.tolist(),
  "output": actual.tolist(),
  "relative_error": err,
  "tolerance": 1e-5,
  "row_contents_row0": vals[0].tolist(),
  "shape_dtype_finite_unmodified": structural,
  "passes_contract": bool(structural and err <= 1e-5)
}))