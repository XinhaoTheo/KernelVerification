
import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_101/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs("cuda")
out = k.run(a, b)
af = a.cpu().numpy().astype(np.float64); bf = b.cpu().numpy().astype(np.float64)
T, D = af.shape
ref = np.empty_like(af); h = np.zeros(D)
for t in range(T):
    h = af[t]*h + bf[t]; ref[t] = h
o = out.cpu().numpy().astype(np.float64)
diff = o - ref
num = np.linalg.norm(diff.ravel())
den = max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(ref.size))
E = num/den
print(json.dumps({"E": float(E), "norm_out_ref_diff": float(num), "norm_ref": float(np.linalg.norm(ref.ravel())), "num_elements": int(ref.size), "tolerance": 0.003, "all_finite": bool(np.isfinite(o).all()), "max_abs_err": float(np.abs(diff).max())}))
