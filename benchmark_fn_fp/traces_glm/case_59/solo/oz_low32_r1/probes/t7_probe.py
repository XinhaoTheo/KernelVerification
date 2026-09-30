import sys, json
sys.path.insert(0, "/root/numerical_cases/case_x")
import kernel
m = kernel.make_inputs("cuda")[0]
out = kernel.run(m)
ref = float(np.log(np.linalg.det(m.cpu().numpy().astype(np.float64)))) if False else None
import numpy as np
ref = float(np.log(np.linalg.det(m.cpu().numpy().astype(np.float64))))
val = float(out.cpu().numpy()[0])
err = abs(val - ref) / max(abs(ref), 1.0)
print(json.dumps({"kernel": val, "ref": ref, "rel_err": err, "pass": err <= 1e-4, "shape": tuple(out.shape), "dtype": str(out.dtype), "finite": bool(np.isfinite(val))}))