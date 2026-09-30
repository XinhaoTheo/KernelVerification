import json
import numpy as np
import torch

import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e02/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

x, y, q = k.make_inputs()
out = k.run(x, y, q)

# float64 contract oracle: centered ridge, explicit normal equations
X = x.cpu().numpy().astype(np.float64)
Y = y.cpu().numpy().astype(np.float64)
Q = q.cpu().numpy().astype(np.float64)
mu_x = X.mean(axis=0); mu_y = Y.mean()
Xc = X - mu_x; Yc = Y - mu_y
A = Xc.T @ Xc / 32 + np.eye(2) / 1024
rhs = Xc.T @ Yc / 32
beta = np.linalg.solve(A, rhs)
pred = mu_y + (Q - mu_x) @ beta

o = out.cpu().numpy().astype(np.float64)
diff = np.linalg.norm(o - pred)
denom = max(np.linalg.norm(pred), 0.1)
rel = diff / denom

# also: fp32 raw-moment diagnostics
s00 = np.float32(0)
for i in range(32):
    x0 = np.float32(x.cpu().numpy()[i,0]); x1 = np.float32(x.cpu().numpy()[i,1])
    s00 += x0*x0
m0 = np.float32(x.cpu().numpy()[:,0].mean())
cov00_fp32 = np.float64(s00/32 - m0*m0)
cov00_exact = np.float64(Xc[:,0].var())

print(json.dumps({
    "output": o.tolist(),
    "pred_f64": pred.tolist(),
    "beta_f64": beta.tolist(),
    "relative_error": rel,
    "tolerance": 0.05,
    "passed": bool(rel <= 0.05),
    "pred_norm": float(np.linalg.norm(pred)),
    "cov00_fp32": cov00_fp32,
    "cov00_exact": cov00_exact,
    "cov00_rel_err": abs(cov00_fp32-cov00_exact)/cov00_exact
}))