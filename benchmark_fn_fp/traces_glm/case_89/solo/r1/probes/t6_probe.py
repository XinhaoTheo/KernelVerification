import torch, numpy as np, json, sys
sys.path.insert(0, '/root/pilot_cases/case_89')
import kernel as K

a, b = K.make_inputs()
out = K.run(a, b)
a64, b64 = a.double().cpu().numpy(), b.double().cpu().numpy()
ref = np.empty_like(a64)
h = np.zeros(a64.shape[1])
for t in range(a64.shape[0]):
    h = a64[t]*h + b64[t]
    ref[t] = h
o = out.double().cpu().numpy().ravel(); r = ref.ravel()
E = np.linalg.norm(o - r) / max(np.linalg.norm(r), 0.001*np.sqrt(r.size))
print(json.dumps({"E": float(E), "budget": 0.003, "finite": bool(np.isfinite(o).all()), "ref_norm": float(np.linalg.norm(r)), "max_abs_err": float(np.abs(o-r).max())}))