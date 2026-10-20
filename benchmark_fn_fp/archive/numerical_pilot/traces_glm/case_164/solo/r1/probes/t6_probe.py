import torch, numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_101/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs("cuda")
out = k.run(a, b)
# float64 reference on same float32 inputs
af = a.cpu().numpy().astype(np.float64); bf = b.cpu().numpy().astype(np.float64)
ref = np.empty_like(af)
h = np.zeros(af.shape[1])
for t in range(af.shape[0]):
    h = af[t]*h + bf[t]
    ref[t] = h
o = out.cpu().numpy().astype(np.float64)
diff = o - ref
E = np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(ref.size))
finite = bool(np.isfinite(o).all())
print(np.sqrt(ref.size)*0.001)
print({"E": float(E), "finite": finite, "max_abs_err": float(np.abs(diff).max()),
       "ref_norm": float(np.linalg.norm(ref.ravel())), "budget": 0.003})
