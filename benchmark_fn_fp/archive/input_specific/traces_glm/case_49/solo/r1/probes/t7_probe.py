import json, torch, numpy as np, sys
sys.path.insert(0, "/root/numerical_cases/case_n")
import kernel

coeff, pts = kernel.make_inputs("cuda")
out = kernel.run(coeff, pts)
torch.cuda.synchronize()
out = out.cpu().numpy()

c64 = coeff.cpu().numpy().astype(np.float64)
p64 = pts.cpu().numpy().astype(np.float64)
ref = np.array([sum(c64[i,k]*p64[i]**k for k in range(49)) for i in range(8)])

denom = max(np.linalg.norm(ref), 0.001*np.sqrt(8))
err = np.linalg.norm(out.astype(np.float64)-ref)/denom
print(json.dumps({
  "metric": "relative L2 error vs float64 reference",
  "output": out.tolist(),
  "reference": ref.tolist(),
  "reference_norm": float(np.linalg.norm(ref)),
  "rel_l2_error": float(err),
  "contract_limit": 0.0002,
  "passes": bool(err <= 0.0002 and np.all(np.isfinite(out)))
}))
