import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_k")
import kernel

matrix, rhs = kernel.make_inputs("cuda")
out = kernel.run(matrix, rhs)
out_np = out.detach().cpu().numpy().astype(np.float64)

A = matrix.detach().cpu().numpy().astype(np.float64)
b = rhs.detach().cpu().numpy().astype(np.float64)
x = np.linalg.solve(A, b)

err = np.linalg.norm(out_np - x) / max(np.linalg.norm(x), 0.001*np.sqrt(16))
# modal decomposition for diagnosis
Q = np.array([[1.0]], dtype=np.float64)
while len(Q) < 16:
    Q = np.block([[Q, Q], [Q, -Q]])
Q /= 4.0
lam = np.array([0.03125,0.0625,0.125,0.25,0.375,0.5,0.625,0.75,
                0.875,1.0,1.125,1.25,1.375,1.5,1.625,1.75])
bm = Q.T @ b
xm = bm / lam
errm = Q.T @ (out_np - x)
print(json.dumps({
    "metric": "relative L2 error vs fp64 exact solve",
    "relative_l2_error": float(err),
    "tolerance": 0.08,
    "norm_x": float(np.linalg.norm(x)),
    "finite": bool(np.all(np.isfinite(out_np))),
    "shape": list(out.shape),
    "per_mode_truncation_factor": [float((1-l)**64) for l in lam[:3]],
    "b_modal_min_lambda": float(bm[np.argmin(lam)]),
    "x_modal_min_lambda": float(xm[np.argmin(lam)]),
    "err_modal_min_lambda": float(errm[np.argmin(lam)]),
    "max_modal_err": float(np.max(np.abs(errm))),
    "passes_contract": bool(err <= 0.08)
}))
