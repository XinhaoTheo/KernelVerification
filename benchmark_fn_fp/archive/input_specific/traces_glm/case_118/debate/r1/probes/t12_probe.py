import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_t/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, f, p = k.make_inputs_numpy()
A = a.astype(np.float64); F = f.astype(np.float64); P = p.astype(np.float64)
# float64 closed-form reference
ref = 1.0 + np.sum(A*(np.cos(P)-np.cos(P+F))/F, axis=1)
# exact float64 32-point midpoint rule (emulates kernel's math in high precision)
pts = (np.arange(32)+0.5)/32.0
mid = 1.0 + np.sum(np.sum(A*np.sin(F[None,:,:]*pts[:,None,None]+P[None,:,:],axis=2),axis=1),axis=0)/32.0
mid = 1.0 + (A[:,None,:]*np.sin(F[:,None,:]*pts[None,:,None]+P[:,None,:])).sum(axis=(1,2))/32.0
rel_mid = np.linalg.norm(mid-ref)/max(np.linalg.norm(ref),1e-12)
# actual kernel run
try:
    out = k.run(*k.make_inputs()).cpu().numpy()
    rel_k = np.linalg.norm(out.astype(np.float64)-ref)/max(np.linalg.norm(ref),1e-12)
    rel_k_mid = np.linalg.norm(out.astype(np.float64)-mid)/max(np.linalg.norm(mid),1e-12)
    finite = bool(np.all(np.isfinite(out)))
except Exception as e:
    out = None; rel_k = None; rel_k_mid = None; finite = None; print("kernel error:", e)
print(json.dumps({"reference": ref.tolist(), "midpoint_f64": mid.tolist(),
 "kernel_out": None if out is None else out.tolist(),
 "rel_err_midpoint_vs_ref": float(rel_mid), "rel_err_kernel_vs_ref": None if rel_k is None else float(rel_k),
 "rel_err_kernel_vs_midpoint": None if rel_k_mid is None else float(rel_k_mid),
 "tolerance": 0.035, "kernel_finite": finite,
 "frequencies": F.tolist()}))