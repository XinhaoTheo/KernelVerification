import numpy as np, torch, json, math, sys
sys.path.insert(0, "/root/numerical_cases/case_z")
from kernel import run, make_inputs_numpy

query_np, anchors_np, values_np = make_inputs_numpy()
q = torch.from_numpy(query_np).cuda()
a = torch.from_numpy(anchors_np).cuda()
v = torch.from_numpy(values_np).cuda()

out = run(q, a, v)
out_val = float(out.cpu().numpy()[0])

# float64 reference from stored float32 values
q64 = query_np.astype(np.float64); a64 = anchors_np.astype(np.float64); v64 = values_np.astype(np.float64)
d = ((a64 - q64)**2).sum(axis=1)
w = np.exp(-16.0 * d)
ref = float((w * v64).sum() / w.sum())

err = abs(out_val - ref) / max(abs(ref), 0.05)
print(json.dumps({
  "metric": "contract error |output-ref|/max(|ref|,0.05)",
  "output": out_val, "reference": ref,
  "abs_diff": abs(out_val - ref),
  "error": err, "bound": 0.05, "passes": err <= 0.05,
  "finite": math.isfinite(out_val), "shape": list(out.shape),
  "distances_min_max": [float(d.min()), float(d.max())],
  "weights_min_max": [float(w.min()), float(w.max())]
}))