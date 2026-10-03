import json, sys, os
sys.path.insert(0, '/root/pilot_cases/case_93')
import numpy as np, torch, kernel
x, w = kernel.make_inputs("cuda")
out = kernel.run(x, w).cpu().numpy().astype(np.float64)
ref = w.cpu().numpy().astype(np.float64) @ x.cpu().numpy().astype(np.float64)
n = out.size
E = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(n))
print(json.dumps({"E": E, "finite": bool(np.all(np.isfinite(out))), "ref_norm": float(np.linalg.norm(ref)), "out_norm": float(np.linalg.norm(out))}))