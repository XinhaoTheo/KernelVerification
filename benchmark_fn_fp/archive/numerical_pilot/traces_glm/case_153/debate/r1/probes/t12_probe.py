import sys, json
sys.path.insert(0, "/root/pilot_cases/case_90")
import torch, numpy as np
import kernel as K

a, b = K.make_inputs("cuda")
out = K.run(a, b).cpu()

# float64 reference on same float32 inputs
a64 = a.cpu().numpy().astype(np.float64)
b64 = b.cpu().numpy().astype(np.float64)
T, D = a64.shape
ref = np.empty((T, D), dtype=np.float64)
h = np.zeros(D, dtype=np.float64)
for t in range(T):
    h = a64[t] * h + b64[t]
    ref[t] = h

N = ref.size
err = out.numpy().astype(np.float64).ravel() - ref.ravel()
refn = np.linalg.norm(ref.ravel())
den = max(refn, 0.001 * np.sqrt(N))
E = np.linalg.norm(err) / den
per_t = np.linalg.norm(err.reshape(T, D), axis=1)

print(json.dumps({
    "E": E, "threshold": 0.003, "E_exceeds_budget": bool(E > 0.003),
    "ref_l2": refn, "denominator": den, "N": int(N),
    "max_abs_err": float(np.max(np.abs(err))),
    "max_per_t_err_l2": float(per_t.max()),
    "per_t_err_l2_last": float(per_t[-1]),
    "out_finite": bool(np.isfinite(out.numpy()).all()),
}))