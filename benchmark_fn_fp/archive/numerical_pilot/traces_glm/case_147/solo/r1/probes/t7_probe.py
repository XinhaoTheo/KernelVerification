import json, sys
sys.path.insert(0, "/root/pilot_cases/case_84")
import kernel
import numpy as np, torch

q, k, v = kernel.make_inputs("cuda")
out = kernel.run(q, k, v).cpu().numpy().astype(np.float64)

# reference: float64 on same float32 inputs
q64 = q.cpu().numpy().astype(np.float64)
k64 = k.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
z = k64 @ q64 / np.sqrt(32)
p = np.exp(z - z.max()); p /= p.sum()
ref = p @ v64

err = np.linalg.norm(out - ref)
den = max(np.linalg.norm(ref), 0.001 * np.sqrt(ref.size))
E = err / den
print(json.dumps({"E": E, "err": err, "norm_ref": np.linalg.norm(ref),
                  "finite": bool(np.isfinite(out).all()), "budget": 0.001}))