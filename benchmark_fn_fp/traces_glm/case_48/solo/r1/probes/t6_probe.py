
import sys, json
sys.path.insert(0, "/root/numerical_cases/case_m")
import numpy as np, torch
import kernel

coefficients, points = kernel.make_inputs("cuda")
output = kernel.run(coefficients, points)
torch.cuda.synchronize()
out = output.cpu().numpy().astype(np.float64)

c64 = coefficients.cpu().numpy().astype(np.float64)
p64 = points.cpu().numpy().astype(np.float64)
ref = np.array([sum(c64[i,k] * p64[i]**k for k in range(49)) for i in range(8)])

rel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(8))
print(json.dumps({
    "metric": "relative L2 error vs float64 reference",
    "output": out.tolist(),
    "reference": ref.tolist(),
    "abs_err": (out-ref).tolist(),
    "reference_norm": float(np.linalg.norm(ref)),
    "rel_l2_error": float(rel),
    "passes_contract": bool(rel <= 0.0002),
    "finite": bool(np.all(np.isfinite(out))),
}))
