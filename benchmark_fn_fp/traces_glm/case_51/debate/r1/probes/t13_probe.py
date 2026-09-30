import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_p/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
u, b = k.make_inputs()
out = k.run(u, b)
u64 = u.double().cpu().numpy(); b64 = b.double().cpu().numpy()
alpha = (u64*b64).sum()/(u64*u64).sum()
res = b64 - alpha*u64
ref = res/np.linalg.norm(res)
u32=u.float().cpu().numpy(); b32=b.float().cpu().numpy()
# residual with EXACT (fp64) alpha but fp32 product/subtract rounding
res_exactalpha = np.array([np.float32(np.float32(u32[j]*np.float32(alpha)) and 0) for j in range(32)])
res_exactalpha = np.array([np.float32(b32[j]-np.float32(u32[j]*np.float32(alpha))) for j in range(32)])
nrm = np.linalg.norm(res_exactalpha.astype(np.float64))
out_exactalpha = (res_exactalpha.astype(np.float64)/nrm)
o = out.double().cpu().numpy()
rel_kernel = np.linalg.norm(o-ref)/np.linalg.norm(ref)
rel_exactalpha = np.linalg.norm(out_exactalpha-ref)/np.linalg.norm(ref)
print(json.dumps({
 "rel_l2_kernel": float(rel_kernel),
 "rel_l2_exact_alpha_fp32_res": float(rel_exactalpha),
 "tolerance": 0.01,
 "residual_fp64_rms": float(np.linalg.norm(res)/np.sqrt(32)),
 "res_exactalpha_rms": float(nrm/np.sqrt(32)),
 "max_elem_ratio_err_to_res": float(np.max(np.abs(res_exactalpha.astype(np.float64)-res)/np.maximum(np.abs(res),1e-30)))}))