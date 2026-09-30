import numpy as np, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_l/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
M, b = k.make_inputs_numpy()
A = M.astype(np.float64); rhs = b.astype(np.float64)
x = np.linalg.solve(A, rhs)
state = np.zeros(16)
for _ in range(64):
    state = state + (rhs - A @ state)
den = max(np.linalg.norm(x), 0.001*np.sqrt(16))
err = np.linalg.norm(state - x)/den
Q = np.array([[1.0]], dtype=np.float64)
while len(Q) < 16:
    Q = np.block([[Q, Q], [Q, -Q]])
Q /= 4.0
bmodes = np.abs(Q.T @ rhs)
print(json.dumps({"rel_err_f64_64step": float(err), "tolerance": 0.08, "norm_x": float(np.linalg.norm(x)),
                  "denominator": float(den), "bmode_min_lambda": float(bmodes[0]), "bmode_second": float(bmodes[1])}))