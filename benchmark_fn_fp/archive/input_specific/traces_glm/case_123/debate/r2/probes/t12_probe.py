import numpy as np, json, sys
sys.path.insert(0, "/root/numerical_cases/case_s")
from kernel import make_inputs_numpy, run

a, w, p = make_inputs_numpy()  # float32 seeded arrays
a64, w64, p64 = a.astype(np.float64), w.astype(np.float64), p.astype(np.float64)

# fp64 closed-form reference
ref = 1.0 + (a64 * (np.cos(p64) - np.cos(p64 + w64)) / w64).sum(axis=1)

# fp64 32-point midpoint rule (isolates discretization error)
t = (np.arange(32, dtype=np.float64) + 0.5) / 32.0
val = np.ones((4, 32), dtype=np.float64)
for k in range(8):
    val += a64[:, k, None] * np.sin(w64[:, k, None] * t[None, :] + p64[:, k, None])
mid64 = val.mean(axis=1)

rel_mid = np.linalg.norm(mid64 - ref) / max(np.linalg.norm(ref), 1e-12)

# actual kernel output
import torch
out = run(*make_inputs("cuda")).cpu().numpy().astype(np.float64)
rel_kernel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-12)

print(json.dumps({
    "metric": "relative L2 vs fp64 closed-form reference (seed 711165)",
    "ref_norm": float(np.linalg.norm(ref)),
    "rel_err_fp64_midpoint": float(rel_mid),
    "rel_err_kernel_fp32": float(rel_kernel),
    "tolerance": 0.035,
    "kernel_output": [float(x) for x in out],
    "reference": [float(x) for x in ref],
    "max_freq": float(w64.max()),
    "kernel_passes_contract": bool(rel_kernel <= 0.035),
    "midpoint_fp64_passes": bool(rel_mid <= 0.035),
}))
