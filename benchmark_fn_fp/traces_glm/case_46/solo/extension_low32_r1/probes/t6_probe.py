import numpy as np, torch, sys, os
sys.path.insert(0, "/root/numerical_cases/case_k")
from kernel import make_inputs, run
m, r = make_inputs()
out = run(m, r)
x = np.linalg.solve(m.cpu().numpy().astype(np.float64), r.cpu().numpy().astype(np.float64))
o = out.cpu().numpy().astype(np.float64)
err = np.linalg.norm(o - x) / max(np.linalg.norm(x), 0.001*np.sqrt(16))
print({"metric":"relative_L2_error","value":float(err),"tol":0.08,"finite":bool(np.all(np.isfinite(o))),"shape":list(out.shape),"norm_x":float(np.linalg.norm(x))})
