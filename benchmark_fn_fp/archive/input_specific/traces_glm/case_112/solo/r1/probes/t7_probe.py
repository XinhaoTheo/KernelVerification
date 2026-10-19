
import json, sys, math
sys.path.insert(0, "/root/numerical_cases/case_t")
import numpy as np, torch
from kernel import make_inputs, run

a, f, p = make_inputs()
out = run(a, f, p).cpu().numpy()
A, F, P = a.cpu().numpy().astype(np.float64), f.cpu().numpy().astype(np.float64), p.cpu().numpy().astype(np.float64)
ref = 1.0 + (A*(np.cos(P)-np.cos(P+F))/F)).sum(axis=1)
rel = np.linalg.norm(out.astype(np.float64)-ref)/max(np.linalg.norm(ref),1e-12)
print(json.dumps({"output": out.tolist(), "reference": ref.tolist(),
                  "rel_l2": float(rel), "finite": bool(np.all(np.isfinite(out))),
                  "tolerance": 0.035, "passes": bool(rel<=0.035)}))
