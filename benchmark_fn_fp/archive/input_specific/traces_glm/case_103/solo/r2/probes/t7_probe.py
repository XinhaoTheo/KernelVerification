import numpy as np, torch, sys, os
sys.path.insert(0, "/root/numerical_cases/case_k")
import kernel
m, r = kernel.make_inputs("cuda")
out = kernel.run(m, r)
torch.cuda.synchronize()
o = out.cpu().numpy().astype(np.float64)
M = m.cpu().numpy().astype(np.float64)
b = r.cpu().numpy().astype(np.float64)
x = np.linalg.solve(M, b)
err = np.linalg.norm(o - x)
den = max(np.linalg.norm(x), 0.001*np.sqrt(16))
rel = err/den
print("norm_x", np.linalg.norm(x), "err", err, "rel", rel, "finite", bool(np.all(np.isfinite(o))), "shape", o.shape)
import json
print(json.dumps({"rel_l2": rel, "pass": rel <= 0.08, "finite": bool(np.all(np.isfinite(o)))}))
