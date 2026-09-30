
import numpy as np, torch, sys, json
sys.path.insert(0, "/root/numerical_cases/case_x")
import kernel

m, = kernel.make_inputs(device="cuda")
out = kernel.run(m).cpu().numpy()
# reference in float64 of stored fp32 entries
A = m.cpu().numpy().astype(np.float64)
ref = np.float64(np.linalg.slogdet(A)[1])
err = float(np.linalg.norm(out - np.array([ref])) / max(np.linalg.norm(np.array([ref])), 1.0))
print(json.dumps({"output": out.tolist(), "reference": float(ref), "rel_l2_err": err, "pass": err <= 1e-4, "finite": bool(np.isfinite(out).all())}))
