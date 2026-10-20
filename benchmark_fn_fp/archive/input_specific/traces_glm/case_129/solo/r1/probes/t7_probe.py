import numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_y")
from kernel import make_inputs, run

q, a, v = make_inputs()
out = run(q, a, v)
q64 = q.cpu().numpy().astype(np.float64)
a64 = a.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
d = ((a64 - q64)**2).sum(axis=1)
w = np.exp(-16.0 * d)
ref = (w * v64).sum() / w.sum()
o = out.cpu().item()
err = abs(o - ref) / max(abs(ref), 0.05)
print(np.round(np.array([o, ref, err]), 8).tolist())
import json
print(json.dumps({"output": o, "reference": ref, "normalized_error": err, "finite": bool(np.isfinite(o)), "shape": tuple(out.shape)}))
