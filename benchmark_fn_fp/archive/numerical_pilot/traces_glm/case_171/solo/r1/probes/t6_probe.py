
import sys, json, math
sys.path.insert(0, "/root/pilot_cases/case_102")
import numpy as np, torch
import kernel as K

q, k, v = K.make_inputs("cuda")
out = K.run(q, k, v)

# float64 reference on the SAME float32 input values
q64 = q.double().cpu().numpy(); k64 = k.double().cpu().numpy(); v64 = v.double().cpu().numpy()
z = k64 @ q64 / math.sqrt(32)
z = z - z.max(); p = np.exp(z); p /= p.sum()
ref = p @ v64

o = out.double().cpu().numpy()
numel = o.size
E = np.linalg.norm(o.ravel() - ref.ravel()) / max(np.linalg.norm(ref.ravel()), 0.001*math.sqrt(numel))
print(json.dumps({
  "metric": "E per contract definition",
  "E": float(E),
  "finite": bool(np.isfinite(o).all()),
  "ref_norm": float(np.linalg.norm(ref)),
  "out_norm": float(np.linalg.norm(o)),
  "max_abs_err": float(np.abs(o-ref).max()),
  "shapes": [list(q.shape), list(k.shape), list(v.shape), list(out.shape)],
  "verdict_budget": "pass" if (E <= 0.001 and np.isfinite(o).all()) else "fail"
}))
