
import numpy as np, torch, json, math, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_s/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, f, p = k.make_inputs("cuda")
out = k.run(a, f, p)
A = a.double().cpu().numpy(); F = f.double().cpu().numpy(); P = p.double().cpu().numpy()
ref = 1 + (A*(np.cos(P)-np.cos(P+F))/F).sum(axis=1)
o = out.double().cpu().numpy()
rel = np.linalg.norm(o-ref)/max(np.linalg.norm(ref),1e-12)
print(json.dumps({"output": o.tolist(), "reference": ref.tolist(),
                  "rel_l2": float(rel), "tolerance": 0.035,
                  "within_tol": bool(rel<=0.035),
                  "finite": bool(np.isfinite(o).all())}))
