import numpy as np, torch
rng = np.random.Generator(np.random.PCG64(812))
n, d = 64, 32
q = rng.standard_normal(d)
k = rng.standard_normal((n, d)) * 0.7
v = rng.standard_normal((n, d))
q32, k32 = q.astype(np.float32), k.astype(np.float32)
z = k32.astype(np.float64) @ q32.astype(np.float64) / np.sqrt(d)
p = np.exp(z - z.max()); p /= p.sum()
v -= 0.5 * (p @ v)[None, :]
# float32-rounded inputs exactly as returned by make_inputs
q_f = q32.astype(np.float64); k_f = k32.astype(np.float64); v_f = v.astype(np.float32).astype(np.float64)
# float64 reference on the rounded inputs
z_r = k_f @ q_f / np.sqrt(d)
p_r = np.exp(z_r - z_r.max()); p_r /= p_r.sum()
y_ref = p_r @ v_f
ref_norm = np.linalg.norm(y_ref)
floor = 0.001 * np.sqrt(d)
import json
print(json.dumps({
  "ref_norm": float(ref_norm),
  "floor": float(floor),
  "dominant": "floor" if floor > ref_norm else "ref_norm",
  "ratio_ref_to_floor": float(ref_norm / floor)
}))