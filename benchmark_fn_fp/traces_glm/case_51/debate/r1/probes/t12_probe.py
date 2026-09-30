import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_p/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
u, b = k.make_inputs()
out = k.run(u, b)
u64 = u.double().cpu().numpy(); b64 = b.double().cpu().numpy()
alpha = (u64*b64).sum() / (u64*u64).sum()
res = b64 - alpha*u64
ref = res / np.linalg.norm(res)
o = out.double().cpu().numpy()
rel = np.linalg.norm(o-ref)/max(np.linalg.norm(ref),1e-12)
# residual scale diagnostics
true_res_scale = np.linalg.norm(res)/np.sqrt(32)
# simulate fp32 alpha (sequential accumulation, separately rounded) in numpy
u32=u.float().cpu().numpy(); b32=b.float().cpu().numpy()
num=np.float32(0); den=np.float32(0)
for j in range(32):
    num=np.float32(num+np.float32(u32[j]*b32[j])); den=np.float32(den+np.float32(u32[j]*u32[j]))
alpha32=np.float32(num/den)
print(json.dumps({
 "rel_l2_error": float(rel), "tolerance": 0.01, "passes": bool(rel<=0.01),
 "alpha_fp64": float(alpha), "alpha_fp32_sim": float(alpha32),
 "alpha_abs_err": float(abs(alpha32-alpha)),
 "residual_rms_scale": float(true_res_scale),
 "output_finite": bool(np.isfinite(o).all())}))