
import importlib.util, sys, torch, numpy as np
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_82/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
q, k, v = m.make_inputs("cuda")
out = m.run(q, k, v)
# float64 reference on same float32 values
q64 = q.cpu().numpy().astype(np.float64)
k64 = k.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
z = k64 @ q64 / np.sqrt(q64.shape[0])
z -= z.max(); p = np.exp(z); p /= p.sum()
ref = p @ v64
o = out.cpu().numpy().astype(np.float64)
num = np.linalg.norm(o.ravel() - ref.ravel())
den = max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(ref.size))
print({"E": num/den, "num": num, "den": den, "norm_ref": np.linalg.norm(ref), "finite": bool(np.isfinite(o).all())})
