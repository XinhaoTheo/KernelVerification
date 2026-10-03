import json, importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("kernel", "/root/pilot_cases/case_94/kernel.py")
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
q, k, v = km.make_inputs("cuda")
out = km.run(q, k, v)
qd = q.cpu().numpy().astype(np.float64); kd = k.cpu().numpy().astype(np.float64); vd = v.cpu().numpy().astype(np.float64)
z = kd @ qd / np.sqrt(32)
p = np.exp(z - z.max()); p /= p.sum()
ref = p @ vd
E = np.linalg.norm(out.cpu().numpy() - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(32))
print(json.dumps({"E": float(E), "ref_norm": float(np.linalg.norm(ref)),
 "floor": 0.001*np.sqrt(32), "all_finite": bool(np.isfinite(out.cpu().numpy()).all()),
 "budget": 0.001, "meets_budget": bool(E <= 0.001)}))