import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_e")
import kernel

matrix, initial, drive = kernel.make_inputs()
m64 = matrix.cpu().numpy().astype(np.float64)
i64 = initial.cpu().numpy().astype(np.float64)
d64 = drive.cpu().numpy().astype(np.float64)

state = i64.copy()
for t in range(64):
    state = m64 @ state + d64[t]

out = kernel.run(matrix, initial, drive)
o = out.cpu().numpy().astype(np.float64)

num = np.linalg.norm(o - state)
den = max(np.linalg.norm(state), 0.001 * 4.0)
rel = num / den
print(json.dumps({
    "metric": "relative L2 error vs float64 reference",
    "reason": "contract: rel error <= 0.002 on make_inputs() workload",
    "output_shape": list(o.shape),
    "finite": bool(np.all(np.isfinite(o))),
    "ref_norm": float(np.linalg.norm(state)),
    "abs_err": float(num),
    "rel_err": float(rel),
    "tolerance": 0.002,
    "pass": bool(rel <= 0.002 and np.all(np.isfinite(o)))
}))
