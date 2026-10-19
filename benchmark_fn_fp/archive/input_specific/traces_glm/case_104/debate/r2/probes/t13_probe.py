
import json
import numpy as np
import sys
sys.path.insert(0, "/root/numerical_cases/case_l")
import kernel as K

matrix, rhs = K.make_inputs_numpy()
A32 = matrix.astype(np.float32)
b32 = rhs.astype(np.float32)

def iterate(A, b, dtype):
    state = np.zeros(16, dtype=dtype)
    A_ = A.astype(dtype); b_ = b.astype(dtype)
    for _ in range(64):
        residual = b_ - (A_ * state[None, :]).sum(axis=1, dtype=dtype)
        state = state + residual
    return state

x64 = iterate(matrix, rhs, np.float64)
x32 = iterate(matrix, rhs, np.float32)
x_exact = np.linalg.solve(matrix.astype(np.float64), rhs.astype(np.float64))
denom = max(np.linalg.norm(x_exact), 0.001*np.sqrt(16))

rel32_exact = np.linalg.norm(x32.astype(np.float64) - x_exact)/denom
rel64_exact = np.linalg.norm(x64 - x_exact)/denom
rounding_gap = abs(rel32_exact - rel64_exact)
res = {"rel_fp32_iteration_vs_exact": float(rel32_exact),
       "rel_fp64_iteration_vs_exact": float(rel64_exact),
       "rounding_shift_abs": float(rounding_gap),
       "fp32_rounding_meaningful": bool(rounding_gap > 0.005),
       "tolerance": 0.08,
       "note": "if fp64 64-step iteration is under tolerance but fp32 is over, c2 confirmed; otherwise rebutted"}
print(json.dumps(res))
