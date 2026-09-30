
import sys, json
sys.path.insert(0, '/root/numerical_cases/case_u')
import numpy as np, torch
import kernel

coeff, offset = kernel.make_inputs(device='cuda')
out = kernel.run(coeff, offset)
torch.cuda.synchronize()

c64 = coeff.double().cpu().numpy()
o64 = float(offset.double().cpu().numpy())
n = np.arange(64, dtype=np.float64)
ref = np.full(64, o64)
for k in range(1, 17):
    ref += c64[0, k-1]*np.cos(2*np.pi*k*n/64) + c64[1, k-1]*np.sin(2*np.pi*k*n/64)

out_np = out.float().cpu().numpy()
diff = out_np.astype(np.float64) - ref
rel = np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
    "metric": "relative L2 error vs float64 16-mode reference",
    "rel_l2": float(rel),
    "tolerance": 0.15,
    "pass": bool(rel <= 0.15),
    "shape": list(out_np.shape),
    "dtype": str(out.dtype),
    "finite": bool(np.isfinite(out_np).all()),
    "ref_norm": float(np.linalg.norm(ref)),
    "max_abs_err": float(np.abs(diff).max()),
}))
