import json, sys
sys.path.insert(0, "/root/pilot_cases/case_89")
import kernel, torch, numpy as np

a, b = kernel.make_inputs()
out = kernel.run(a, b).double().cpu().numpy()

# fp64 reference on same fp32 inputs
a64 = a.double().cpu().numpy(); b64 = b.double().cpu().numpy()
ref = np.empty_like(b64)
h = np.zeros(b64.shape[1])
for t in range(a64.shape[0]):
    h = a64[t] * h + b64[t]
    ref[t] = h

err = out - ref
N = out.size
ref_norm = np.linalg.norm(ref)
denom = max(ref_norm, 0.001 * np.sqrt(N))
E = np.linalg.norm(err) / denom
finite = bool(np.isfinite(out).all())
per_t = np.linalg.norm(err, axis=1)
print(json.dumps({
    "metric": "E = ||out-ref||_2 / max(||ref||_2, 0.001*sqrt(N)), fp64 reference on exact seeded workload",
    "E": E, "tolerance": 0.003, "budget_violation": bool(E > 0.003),
    "ref_norm": ref_norm, "denom": denom, "err_norm": np.linalg.norm(err),
    "err_rms": float(np.sqrt((err**2).mean())), "err_max_abs": float(np.abs(err).max()),
    "finite": finite, "N": N,
    "err_norm_last_quarter": float(per_t[384:].sum()),
    "peak_h": float(ref.max()), "per_t_err_t511": float(per_t[-1]),
}))