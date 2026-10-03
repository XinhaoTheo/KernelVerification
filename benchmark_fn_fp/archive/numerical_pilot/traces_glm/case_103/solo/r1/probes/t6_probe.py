import torch, numpy as np, json, sys
sys.path.insert(0, "/root/pilot_cases/case_103")
import kernel as K

q, k, v = K.make_inputs("cuda")
out = K.run(q, k, v).double().cpu().numpy()

# float64 reference on same float32 inputs
q64, k64, v64 = q.double(), k.double(), v.double()
z = k64 @ q64 / np.sqrt(32.0, dtype=np.float64)
p = np.exp(z - z.max()); p /= p.sum()
ref = (p[:, None] * v64.cpu().numpy()).sum(0) if False else (p[:, None] * v64.cpu().numpy()).sum(0)

out32 = out.astype(np.float64)
num = np.linalg.norm(out32 - ref)
den = max(np.linalg.norm(ref), 0.001 * np.sqrt(ref.size))
print(json.dumps({
  "E": num / den,
  "num": float(num), "den": float(den),
  "finite": bool(np.all(np.isfinite(out32))),
  "ref_norm": float(np.linalg.norm(ref)),
  "max_abs_err": float(np.max(np.abs(out32 - ref))),
}))
