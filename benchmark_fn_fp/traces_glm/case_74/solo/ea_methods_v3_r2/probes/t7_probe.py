import numpy as np, torch, json, sys
sys.path.insert(0, "/root/evidence_cases/case_e13")
import kernel

x, = kernel.make_inputs("cuda")
before = x.clone()
out = kernel.run(x)
# exact target: stored float32 values summed exactly in float64 (sum of low values; 2^80 pair and 2^30 pair cancel exactly)
vals = before.cpu().numpy().astype(np.float64)
target = vals.sum(axis=1)  # float64 exact for these magnitudes
actual = out.cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(actual - target) / max(np.linalg.norm(target), 1e-12))
print(json.dumps({
  "metric": "relative 2-norm error vs exact float64 row sums",
  "target": target.tolist(),
  "output": actual.tolist(),
  "relative_error": rel,
  "tolerance": 1e-5,
  "passed_metric": rel <= 1e-5,
  "structural": (tuple(out.shape)==(4,) and out.dtype==torch.float32
                 and bool(np.isfinite(actual).all()) and bool(torch.equal(x, before)))
}))