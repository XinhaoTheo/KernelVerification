import json, numpy as np, torch
import sys
sys.path.insert(0, '/root/numerical_cases/case_u')
from kernel import make_inputs, run, make_inputs_numpy

coef, offset = make_inputs()
out = run(coef, offset).cpu().numpy()

c64 = make_inputs_numpy()[0].astype(np.float64)
off64 = float(make_inputs_numpy()[1][0])
n = np.arange(64, dtype=np.float64)
ref = off64 * np.ones(64)
for k in range(1, 17):
    ref += c64[0, k-1] * np.cos(2*np.pi*k*n/64) + c64[1, k-1] * np.sin(2*np.pi*k*n/64)

err = np.linalg.norm(out.astype(np.float64) - ref)
ratio = err / max(np.linalg.norm(ref), 1e-12)
# pure truncation error (retained modes in fp64)
trunc = np.zeros(64)
for k in range(7, 17):
    trunc += c64[0, k-1] * np.cos(2*np.pi*k*n/64) + c64[1, k-1] * np.sin(2*np.pi*k*n/64)
trunc_ratio = np.linalg.norm(trunc) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
    "metric": "relative L2 error vs float64 all-16-mode reference (fixed seeded workload)",
    "ratio_total": float(ratio),
    "ratio_truncation_only": float(trunc_ratio),
    "tolerance": 0.15,
    "passes_contract": bool(ratio <= 0.15),
    "finite": bool(np.all(np.isfinite(out))),
    "shape": list(out.shape),
    "ref_norm": float(np.linalg.norm(ref)),
}))
