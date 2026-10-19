import numpy as np, torch, json
from fractions import Fraction
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e15/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, = k.make_inputs()
before = x.clone()
out = k.run(x)
out64 = out.detach().cpu().numpy().astype(np.float64)
xr = x.detach().cpu().numpy()
target = np.array([float(sum(Fraction(float(v)) for v in xr[r])) for r in range(4)])
err = float(np.linalg.norm(out64 - target)/max(np.linalg.norm(target),1e-12))
print(json.dumps({"target": target.tolist(), "output": out64.tolist(),
 "relative_error": err, "tolerance": 1e-5,
 "unmodified": bool(torch.equal(x,before)),
 "finite": bool(np.isfinite(out64).all()),
 "passes": bool(err <= 1e-5)}))