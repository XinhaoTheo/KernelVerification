
import json, sys, os
sys.path.insert(0, "/root/numerical_cases/case_y")
import numpy as np, torch
import kernel as K

q, a, v = K.make_inputs("cuda")
out = K.run(q, a, v)
torch.cuda.synchronize()

q64 = q.cpu().numpy().astype(np.float64)
a64 = a.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
d = ((a64 - q64)**2).sum(axis=1)
w = np.exp(-16.0 * d)
ref = (w * v64).sum() / w.sum()

o = float(out.item())
err = abs(o - ref) / max(abs(ref), 0.05)
print(json.dumps({
    "kernel_output": o, "reference": ref, "abs_diff": o - ref,
    "relative_error": err, "threshold": 0.05, "finite": bool(np.isfinite(o)),
    "distances_min_max": [float(d.min()), float(d.max())]
}))
