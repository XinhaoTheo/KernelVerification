import torch, numpy as np, json, sys
sys.path.insert(0, "/root/pilot_cases/case_97")
import kernel
a, b = kernel.make_inputs()
out = kernel.run(a, b)
out = out.float().cpu().numpy()
af = a.cpu().numpy().astype(np.float64)
bf = b.cpu().numpy().astype(np.float64)
# reference in float64 on same float32 inputs
ref = np.empty_like(bf)
h = np.zeros(bf.shape[1])
for t in range(bf.shape[0]):
    h = af[t]*h + bf[t]
    ref[t] = h
n = ref.size
err = np.linalg.norm((out-ref).ravel())
den = max(np.linalg.norm(ref.ravel()), 0.001*np.sqrt(n))
E = err/den
print(json.dumps({"E": float(E), "err": float(err), "ref_norm": float(np.linalg.norm(ref.ravel())), "n": int(n), "finite": bool(np.isfinite(out).all())}))