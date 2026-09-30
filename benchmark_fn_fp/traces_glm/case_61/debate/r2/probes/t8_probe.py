import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_z")
import kernel

q, a, v = kernel.make_inputs_numpy()
qt, at, vt = kernel.make_inputs()
out = kernel.run(qt, at, vt).detach().cpu().numpy()

# float64 reference from stored float32 values
d64 = ((a.astype(np.float64) - q.astype(np.float64)[None, :])**2).sum(axis=1)
w64 = np.exp(-16.0*d64)
ref = (w64*v.astype(np.float64)).sum()/w64.sum()

err = abs(float(out[0]) - float(ref))/max(abs(float(ref)), 0.05)
print(json.dumps({
  "kernel_output": float(out[0]), "reference": float(ref),
  "relative_error": err, "tolerance": 0.05, "passes": err <= 0.05,
  "finiteness": bool(np.isfinite(out).all()), "shape": list(out.shape),
  "true_distances_min_max": [float(d64.min()), float(d64.max())]
}))