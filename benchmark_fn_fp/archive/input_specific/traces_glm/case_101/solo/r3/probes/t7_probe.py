import sys, json
sys.path.insert(0, "/root/numerical_cases/case_c")
import numpy as np, torch
from kernel import make_inputs, run

logits, values = make_inputs()
out = run(logits, values)
# float64 reference on supplied float32 values
l64 = logits.to(torch.float64).cpu().numpy()
v64 = values.to(torch.float64).cpu().numpy()
p = np.exp(l64 - l64.max()); p /= p.sum()
ref = p @ v64
o = out.cpu().numpy().astype(np.float64)
rel = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(16))
print(json.dumps({"rel_l2": float(rel), "finite": bool(np.all(np.isfinite(o))),
                  "shape": list(out.shape), "tol": 0.02, "ref_norm": float(np.linalg.norm(ref))}))
