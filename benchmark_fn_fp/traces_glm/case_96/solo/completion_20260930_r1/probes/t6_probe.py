import json, torch, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_96/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
q, k, v = m.make_inputs()
out = m.run(q, k, v).cpu().numpy().astype(np.float64)
q64, k64, v64 = q.cpu().numpy().astype(np.float64), k.cpu().numpy().astype(np.float64), v.cpu().numpy().astype(np.float64)
z = k64 @ q64 / np.sqrt(k64.shape[1])
p = np.exp(z - z.max()); p /= p.sum()
ref = p @ v64
E = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
print(json.dumps({"E": float(E), "finite": bool(np.all(np.isfinite(out))), "ref_norm": float(np.linalg.norm(ref)), "out_norm": float(np.linalg.norm(out)), "max_abs_err": float(np.max(np.abs(out-ref)))}))