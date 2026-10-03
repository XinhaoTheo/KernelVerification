import json, torch, numpy as np, sys
sys.path.insert(0, "/root/pilot_cases/case_100")
import kernel
x, w = kernel.make_inputs()
out = kernel.run(x, w)
ref = w.cpu().numpy().astype(np.float64) @ x.cpu().numpy().astype(np.float64)
o = out.cpu().numpy().astype(np.float64)
num = o.size
E = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(num))
print(json.dumps({"E": float(E), "finite": bool(np.isfinite(o).all()),
  "ref_norm": float(np.linalg.norm(ref)), "err_norm": float(np.linalg.norm(o-ref))}))