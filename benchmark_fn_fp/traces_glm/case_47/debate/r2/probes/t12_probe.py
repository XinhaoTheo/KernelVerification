
import json
import numpy as np
import torch
import sys
sys.path.insert(0, "/root/numerical_cases/case_l")
import kernel as K

matrix, rhs = K.make_inputs()
out = K.run(matrix, rhs)
out_np = out.detach().cpu().numpy().astype(np.float64)

A = matrix.cpu().numpy().astype(np.float64)
b = rhs.cpu().numpy().astype(np.float64)
x = np.linalg.solve(A, b)

denom = max(np.linalg.norm(x), 0.001*np.sqrt(16))
rel = np.linalg.norm(out_np - x) / denom

# per-mode diagnostics
w, V = np.linalg.eigh(A)
c = V.T @ b
# error in eigenbasis
err_e = V.T @ (out_np - x)
x_e = V.T @ x
res = {"rel_l2_error": float(rel),
       "tolerance": 0.08,
       "exceeds_tolerance": bool(rel > 0.08),
       "norm_x": float(np.linalg.norm(x)),
       "c_min_abs": float(np.abs(c)[0]),
       "c_second_abs": float(np.abs(c)[1]),
       "err_eig_min": float(err_e[0]),
       "x_eig_min": float(x_e[0]),
       "lambdas_first3": [float(v) for v in w[:3]],
       "output_finite": bool(np.all(np.isfinite(out_np)))}
print(json.dumps(res))
