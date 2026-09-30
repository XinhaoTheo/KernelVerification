import torch, json, numpy as np, sys, os
sys.path.insert(0, "/root/pilot_cases/case_98")
import kernel as K

a, b = K.make_inputs("cuda")
out = K.run(a, b)

a64 = a.double().cpu().numpy(); b64 = b.double().cpu().numpy()
ref = np.zeros_like(b64); h = np.zeros(b64.shape[1])
for t in range(a64.shape[0]):
    h = a64[t] * h + b64[t]
    ref[t] = h

o = out.float().cpu().numpy()
diff = o - ref
num = float(np.linalg.norm(diff.ravel()))
den = max(float(np.linalg.norm(ref.ravel())), 0.001*np.sqrt(ref.size))
E = num/den
res = {
    "metric": "E = ||out-ref||_2 / max(||ref||_2, 0.001*sqrt(N)) on exact workload",
    "E": E, "budget": 0.003,
    "num_norm": num, "ref_norm": float(np.linalg.norm(ref.ravel())),
    "all_finite": bool(np.isfinite(o).all()),
    "max_abs_err": float(np.abs(diff).max()),
    "max_rel_err": float((np.abs(diff)/np.abs(ref)).max()),
    "exceeds_budget": bool(E > 0.003),
}
print(json.dumps(res))