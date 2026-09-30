import torch, numpy as np, json, sys, math
sys.path.insert(0, "/root/pilot_cases/case_103")
import kernel as K

q, k, v = K.make_inputs("cuda")
out = K.run(q, k, v).cpu().double().numpy()
assert np.all(np.isfinite(out))

# float64 reference on same float32 input values
q64 = q.cpu().double().numpy()
k64 = k.cpu().double().numpy()
v64 = v.cpu().double().numpy()
z = k64 @ q64 / math.sqrt(32.0)
z = z - z.max()
p = np.exp(z); p /= p.sum()
ref = p @ v64

num = np.linalg.norm(out - ref)
den = max(np.linalg.norm(ref), 0.001 * np.sqrt(ref.size))
print(json.dumps({
  "E": float(num / den),
  "budget": 0.001,
  "num": float(num), "den": float(den),
  "ref_norm": float(np.linalg.norm(ref)),
  "max_abs_err": float(np.max(np.abs(out - ref))),
  "finite": bool(np.all(np.isfinite(out))),
  "shape": list(out.shape),
}))
