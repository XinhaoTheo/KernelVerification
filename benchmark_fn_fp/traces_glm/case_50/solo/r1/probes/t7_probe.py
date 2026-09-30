
import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_o")
from kernel import run, make_inputs

u, b = make_inputs()
out = run(u, b)

# float64 reference (recentring to avoid cancellation in the reference itself)
u64 = u.cpu().numpy().astype(np.float64)
b64 = b.cpu().numpy().astype(np.float64)
# reference: b = 1.125u + eps stored in fp32; compute alpha exactly from stored values
alpha = (u64*b64).sum() / (u64*u64).sum()
res = b64 - alpha*u64
# alternative recentred: use dot of (b - mean-shift) — but straightforward fp64 is fine here
ref = res / np.linalg.norm(res)

o = out.cpu().numpy().astype(np.float64)
rel = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
  "relative_l2": float(rel),
  "tolerance": 0.01,
  "passes": bool(rel <= 0.01),
  "finite": bool(np.all(np.isfinite(o))),
  "shape": list(o.shape),
  "norm_res": float(np.linalg.norm(res)),
  "norm_b": float(np.linalg.norm(b64)),
  "max_abs_err": float(np.abs(o-ref).max()),
}))
