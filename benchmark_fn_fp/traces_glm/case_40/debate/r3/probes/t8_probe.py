import json, sys
sys.path.insert(0, "/root/numerical_cases/case_e")
import numpy as np, torch
import kernel

matrix, initial, drive = kernel.make_inputs()
out = kernel.run(matrix, initial, drive)
out = out.detach().cpu().numpy().astype(np.float64)

m = matrix.detach().cpu().numpy().astype(np.float64)
x = initial.detach().cpu().numpy().astype(np.float64)
d = drive.detach().cpu().numpy().astype(np.float64)
for t in range(64):
    x = m @ x + d[t]

ref_norm = np.linalg.norm(x)
denom = max(ref_norm, 0.001 * np.sqrt(16))
err = np.linalg.norm(out - x) / denom
print(json.dumps({
    "rel_l2_error": float(err),
    "tolerance": 0.002,
    "passes": bool(err <= 0.002),
    "ref_norm": float(ref_norm),
    "out_finite": bool(np.all(np.isfinite(out))),
    "max_abs_err": float(np.max(np.abs(out - x))),
    "dtype": "float32", "shape": list(out.shape)
}))