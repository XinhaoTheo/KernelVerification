import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_y")
from kernel import run, make_inputs

query, anchors, values = make_inputs()
out = run(query, anchors, values).item()

q64 = query.numpy().astype(np.float64)
a64 = anchors.numpy().astype(np.float64)
v64 = values.numpy().astype(np.float64)
d = ((a64 - q64)**2).sum(axis=1)
w = np.exp(-16.0 * d)
ref = (w * v64).sum() / w.sum()

metric = abs(out - ref) / max(abs(ref), 0.05)
print(json.dumps({
    "kernel_output": out, "reference": float(ref),
    "abs_err": float(abs(out - ref)),
    "error_metric": float(metric),
    "tolerance": 0.05,
    "passes": bool(metric <= 0.05),
    "output_finite": bool(np.isfinite(out)),
    "true_dist_range": [float(d.min()), float(d.max())],
    "ref_weight_range": [float(w.min()), float(w.max())],
}))