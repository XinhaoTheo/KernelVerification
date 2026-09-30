import sys, json
sys.path.insert(0, "/root/numerical_cases/case_d")
import numpy as np, torch
from kernel import run, make_inputs_numpy, make_inputs

logits_np, values_np = make_inputs_numpy()
out = run(*make_inputs("cuda")).cpu().numpy().astype(np.float64)

l = logits_np.astype(np.float64); v = values_np.astype(np.float64)
p = np.exp(l - l.max()); p /= p.sum()
ref = p @ v

rel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 0.001*4.0)
print(json.dumps({"rel_l2_error": rel, "tolerance": 0.02, "finite": bool(np.all(np.isfinite(out))), "shape": list(out.shape), "ref_norm": float(np.linalg.norm(ref)), "out_norm": float(np.linalg.norm(out))}))
