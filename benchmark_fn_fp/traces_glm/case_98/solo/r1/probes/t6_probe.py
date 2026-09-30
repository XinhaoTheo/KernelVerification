import torch, numpy as np, json, sys
sys.path.insert(0, "/root/pilot_cases/case_98")
import kernel as K
a, b = K.make_inputs()
out = K.run(a, b)
# reference in float64 on same float32 inputs
af = a.double().cpu().numpy(); bf = b.double().cpu().numpy()
h = np.zeros_like(bf[0])
ref = np.empty_like(bf)
for t in range(bf.shape[0]):
    h = af[t]*h + bf[t]; ref[t] = h
o = out.float().cpu().numpy()
num = np.linalg.norm(o - ref)
den = max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
finite = bool(np.isfinite(o).all())
E = num/den
print(json.dumps({"E": E, "finite": finite, "num": float(num), "den": float(den), "ref_norm": float(np.linalg.norm(ref)), "max_abs_err": float(np.abs(o-ref).max()), "passes": bool(finite and E<=0.003)}))