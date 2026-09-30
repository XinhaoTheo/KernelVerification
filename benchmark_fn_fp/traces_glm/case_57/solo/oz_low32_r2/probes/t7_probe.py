import numpy as np, torch, json, sys
sys.path.insert(0, '/root/numerical_cases/case_v')
from kernel import run, make_inputs_numpy, make_inputs

coeff_np, off_np = make_inputs_numpy()
coeff, off = make_inputs()
out = run(coeff, off)

n = np.arange(64)
c64 = coeff_np.astype(np.float64)
ref = off_np[0].astype(np.float64)
for k in range(1, 17):
    ref = ref + c64[0, k-1]*np.cos(2*np.pi*k*n/64) + c64[1, k-1]*np.sin(2*np.pi*k*n/64)

o = out.cpu().numpy().astype(np.float64)
rel = np.linalg.norm(o-ref)/max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
  "metric": "relative L2 error vs 16-mode float64 reference",
  "rel_l2": float(rel),
  "tolerance": 0.15,
  "passes": bool(rel <= 0.15),
  "norm_ref": float(np.linalg.norm(ref)),
  "norm_err": float(np.linalg.norm(o-ref)),
  "finite": bool(np.all(np.isfinite(o))),
  "dtype": str(out.dtype), "shape": list(out.shape)
}))
