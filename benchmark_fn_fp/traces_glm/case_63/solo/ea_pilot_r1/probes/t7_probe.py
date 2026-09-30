import json, numpy as np, torch, sys
sys.path.insert(0, "/root/evidence_cases/case_e02")
import kernel

x, y, q = kernel.make_inputs()
out = kernel.run(x, y, q)

# float64 reference over the stored float32 inputs
X = x.double().cpu().numpy(); Y = y.double().cpu().numpy(); Q = q.double().cpu().numpy()
mux = X.mean(axis=0); muy = Y.mean()
Xc = X - mux; yc = Y - muy
lam = 1.0/1024
A = Xc.T @ Xc / 32.0 + lam*np.eye(2)
rhs = Xc.T @ yc / 32.0
beta = np.linalg.solve(A, rhs)
pred = muy + (Q - mux) @ beta
diff = out.double().cpu().numpy() - pred
rel = np.linalg.norm(diff) / max(np.linalg.norm(pred), 0.1)
print(json.dumps({"output": out.cpu().tolist(), "f64_reference": pred.tolist(),
                  "beta": beta.tolist(), "rel_error": float(rel),
                  "tolerance": 0.05, "passed": bool(rel <= 0.05),
                  "finite": bool(np.all(np.isfinite(out.cpu().numpy())))}))