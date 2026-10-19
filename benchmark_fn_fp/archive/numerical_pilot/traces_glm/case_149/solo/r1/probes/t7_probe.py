
import importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("kernel", "/root/pilot_cases/case_92/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs("cuda")
out = k.run(a, b)
af = a.double().cpu().numpy(); bf = b.double().cpu().numpy()
ref = np.empty_like(af)
h = np.zeros(af.shape[1])
for t in range(af.shape[0]):
    h = af[t]*h + bf[t]; ref[t] = h
o = out.double().cpu().numpy()
finite = bool(np.isfinite(o).all())
diff = o - ref
E = np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(ref.size))
print({"finite": finite, "E": float(E), "ref_norm": float(np.linalg.norm(ref)), "max_abs_err": float(np.abs(diff).max()), "shape": o.shape, "dtype": str(out.dtype)})
