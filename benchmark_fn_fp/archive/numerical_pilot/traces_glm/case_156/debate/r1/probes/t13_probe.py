import numpy as np, torch, json, sys
sys.path.insert(0, "/root/pilot_cases/case_99")
import importlib.util
spec = importlib.util.spec_from_file_location("kmod", "/root/pilot_cases/case_99/kernel.py")
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)
q, k, v = kmod.make_inputs("cuda")
out = kmod.run(q, k, v)
torch.cuda.synchronize()
y = out.cpu().numpy().astype(np.float64)
qf = q.cpu().numpy().astype(np.float64)
kf = k.cpu().numpy().astype(np.float64)
vf = v.cpu().numpy().astype(np.float64)
z = kf @ qf / np.sqrt(32)
p = np.exp(z - z.max()); p /= p.sum()
y_ref = p @ vf
ref_norm = np.linalg.norm(y_ref)
floor = 0.001 * np.sqrt(32)
E = np.linalg.norm(y - y_ref) / max(ref_norm, floor)
print(json.dumps({
  "E": float(E),
  "ref_norm": float(ref_norm),
  "floor": float(floor),
  "abs_err_norm": float(np.linalg.norm(y - y_ref)),
  "out_finite": bool(np.isfinite(y).all()),
  "budget": 0.001
}))