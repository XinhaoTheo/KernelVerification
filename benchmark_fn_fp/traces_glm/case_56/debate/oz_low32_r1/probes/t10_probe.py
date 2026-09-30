import sys, json
sys.path.insert(0, "/root/numerical_cases/case_u")
import numpy as np, torch
import kernel

coefficients, offset = kernel.make_inputs_numpy()
c = coefficients.astype(np.float64)
o = float(offset[0])
n = np.arange(64, dtype=np.float64)
ref = o + np.zeros(64)
for k in range(1, 17):
    ref += c[0, k-1]*np.cos(2*np.pi*k*n/64) + c[1, k-1]*np.sin(2*np.pi*k*n/64)

dev = "cuda" if torch.cuda.is_available() else "cpu"
ct = torch.from_numpy(coefficients).to(dev)
ot = torch.from_numpy(offset).to(dev)
out = kernel.run(ct, ot).cpu().numpy().astype(np.float64)

err = np.linalg.norm(out - ref)
nref = np.linalg.norm(ref)
rel = err / max(nref, 1e-12)
dropped_rms = float(np.sqrt(np.mean(c[:, 6:]**2)))
print(json.dumps({
    "relative_l2": rel,
    "abs_err": err,
    "ref_norm": nref,
    "tolerance": 0.15,
    "dropped_mode_rms": dropped_rms,
    "max_abs_err": float(np.max(np.abs(out - ref))),
    "output_finite": bool(np.all(np.isfinite(out))),
    "output_shape": list(out.shape),
    "device": dev,
    "passes_contract": bool(rel <= 0.15)
}))
