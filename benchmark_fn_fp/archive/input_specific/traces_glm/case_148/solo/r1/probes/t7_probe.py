import numpy as np, torch, json, sys
sys.path.insert(0, "/root/evidence_cases/case_e18")
import kernel

x, = kernel.make_inputs()
before = x.clone()
out = kernel.run(x)
xf = x.detach().cpu().numpy().astype(np.float64)
target = xf.sum(axis=1)  # exact since values are exactly representable in f64 and sum is small
actual = out.detach().cpu().numpy().astype(np.float64)
err = float(np.linalg.norm(actual - target) / max(np.linalg.norm(target), 1e-12))
print(json.dumps({
  "metric": "relative L2 error of run(X) vs exact f64 row sums on fixed make_inputs workload",
  "target": target.tolist(),
  "output": actual.tolist(),
  "per_row_abs_error": (actual - target).tolist(),
  "relative_error": err,
  "tolerance": 1e-5,
  "structural": (tuple(out.shape)==(4,) and out.dtype==torch.float32
                 and bool(np.isfinite(actual).all()) and bool(torch.equal(x, before))),
  "passed": bool(err <= 1e-5)
}))
