import torch, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/pilot_cases/case_90/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs("cuda")
out = k.run(a, b)
# reference: float64 recurrence on same float32 inputs
af = a.double().cpu().numpy(); bf = b.double().cpu().numpy()
ref = np.empty_like(bf)
h = np.zeros(bf.shape[1])
for t in range(af.shape[0]):
    h = af[t]*h + bf[t]; ref[t] = h
o = out.double().cpu().numpy()
diff = o - ref
E = np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(ref.size))
print({"E": float(E), "finite": bool(np.isfinite(o).all()),
       "max_abs_err": float(np.abs(diff).max()),
       "ref_norm": float(np.linalg.norm(ref)), "diff_norm": float(np.linalg.norm(diff))})