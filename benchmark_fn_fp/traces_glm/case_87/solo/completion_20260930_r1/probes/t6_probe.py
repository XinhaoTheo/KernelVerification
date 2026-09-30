import sys, json
sys.path.insert(0, '/root/pilot_cases/case_87')
import numpy as np, torch
import kernel as K

a, b = K.make_inputs("cuda")
out = K.run(a, b)

# float64 reference on the same float32 inputs
a64 = a.double().cpu().numpy(); b64 = b.double().cpu().numpy()
ref = np.empty_like(b64)
h = np.zeros(b64.shape[1])
for t in range(b64.shape[0]):
    h = a64[t] * h + b64[t]
    ref[t] = h

o = out.double().cpu().numpy()
N = ref.size
E = np.linalg.norm(o.ravel()-ref.ravel()) / max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(N))
print(json.dumps({"E": float(E), "finite": bool(np.isfinite(o).all()),
                  "ref_norm": float(np.linalg.norm(ref)), "N": N,
                  "max_abs_err": float(np.abs(o-ref).max()),
                  "budget": 0.003}))