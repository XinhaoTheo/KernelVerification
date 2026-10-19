import json, sys
import numpy as np, torch

sys.path.insert(0, "/root/numerical_cases/case_v")
import kernel as K

coeff, off = K.make_inputs("cuda")
out = K.run(coeff, off)

# float64 16-mode reference using supplied float32 values
c = coeff.detach().cpu().numpy().astype(np.float64)
o = float(off.detach().cpu().numpy().astype(np.float64)[0])
n = np.arange(64)
ref = np.full(64, o, dtype=np.float64)
for k in range(1, 17):
    ref += c[0, k-1] * np.cos(2*np.pi*k*n/64) + c[1, k-1] * np.sin(2*np.pi*k*n/64)

o_np = out.detach().cpu().numpy().astype(np.float64)
err = np.linalg.norm(o_np - ref)
denom = max(np.linalg.norm(ref), 1e-12)
ratio = err / denom

# analytic dropped-mode energy (orthogonal basis over 64 samples)
tail = sum(c[0, k-1]**2 + c[1, k-1]**2 for k in range(7, 17))
analytic_ratio = np.sqrt(tail * 32) / denom

print(json.dumps({
    "relative_l2_ratio": float(ratio),
    "tolerance": 0.15,
    "passes_contract": bool(ratio <= 0.15),
    "analytic_truncation_ratio": float(analytic_ratio),
    "err_norm": float(err),
    "ref_norm": float(np.linalg.norm(ref)),
    "output_shape": list(out.shape),
    "output_dtype": str(out.dtype),
    "output_finite": bool(np.isfinite(o_np).all()),
}))
