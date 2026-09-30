import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/numerical_cases/case_t/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

A, F, P = k.make_inputs()
out = k.run(A, F, P).cpu().numpy().astype(np.float64)

a = A.cpu().numpy().astype(np.float64)
f = F.cpu().numpy().astype(np.float64)
p = P.cpu().numpy().astype(np.float64)
t = (np.arange(32)+0.5)/32.0
mid64 = 1.0 + (a[:,None,:]*np.sin(f[:,None,:]*t[None,:,None]+p[:,None,:])).sum(axis=2).mean(axis=1)

ref = 1.0 + np.sum(a*(np.cos(p)-np.cos(p+f))/f, axis=1)
rel_vs_ref = np.linalg.norm(out-ref)/max(np.linalg.norm(ref),1e-12)
diff = out - mid64
rel_fp32 = np.linalg.norm(diff)/max(np.linalg.norm(mid64),1e-12)
quad_err = mid64 - ref
rel_quad = np.linalg.norm(quad_err)/max(np.linalg.norm(ref),1e-12)
print(json.dumps({
 "kernel_minus_midpoint64": diff.tolist(),
 "max_abs_fp32_arith_err": float(np.abs(diff).max()),
 "rel_fp32_component": float(rel_fp32),
 "rel_quadrature_component": float(rel_quad),
 "rel_total_vs_ref": float(rel_vs_ref),
 "tolerance": 0.035,
 "fp32_component_fraction_of_total": float(rel_fp32/rel_vs_ref),
 "fp32_component_decisive": bool(rel_fp32 > 0.01*rel_vs_ref)
}))