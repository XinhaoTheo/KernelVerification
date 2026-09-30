import sys, json, numpy as np, torch
sys.path.insert(0, "/root/pilot_cases/case_93")
import kernel as K

x, w = K.make_inputs("cuda")
out = K.run(x, w)
torch.cuda.synchronize()
out64 = out.double().cpu().numpy()
w64 = w.double().cpu().numpy()
x64 = x.double().cpu().numpy()
ref = w64 @ x64
den = max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
E = np.linalg.norm(out64 - ref) / den
print(json.dumps({
    "metric": "relative_error_E",
    "E": float(E),
    "budget": 0.12,
    "all_finite": bool(np.isfinite(out64).all()),
    "ref_norm": float(np.linalg.norm(ref)),
    "err_norm": float(np.linalg.norm(out64 - ref)),
    "per_row_abs_err_max": float(np.abs(out64-ref).max()),
    "passes": bool(E <= 0.12 and np.isfinite(out64).all())
}))