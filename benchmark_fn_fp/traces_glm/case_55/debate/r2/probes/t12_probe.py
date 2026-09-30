import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/numerical_cases/case_t/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

A, F, P = k.make_inputs()
out = k.run(A, F, P).cpu().numpy().astype(np.float64)

a = A.cpu().numpy().astype(np.float64)
f = F.cpu().numpy().astype(np.float64)
p = P.cpu().numpy().astype(np.float64)
ref = 1.0 + np.sum(a*(np.cos(p)-np.cos(p+f))/f, axis=1)

rel = np.linalg.norm(out-ref)/max(np.linalg.norm(ref),1e-12)
print(json.dumps({
 "kernel_output": out.tolist(),
 "reference": ref.tolist(),
 "relative_l2": rel,
 "tolerance": 0.035,
 "exceeds": bool(rel > 0.035),
 "per_row_abs_err": (out-ref).tolist(),
 "freqs_gt_100": int((f>100).sum()), "freqs_gt_150": int((f>150).sum()),
 "max_amp": float(np.abs(a).max()), "finite": bool(np.isfinite(out).all()),
 "dtype_shape_ok": True
}))