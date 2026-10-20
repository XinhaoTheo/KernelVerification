import numpy as np, torch, json, sys, os
sys.path.insert(0, "/root/pilot_cases/case_104")
os.chdir("/root/pilot_cases/case_104")
import kernel as K

dev = "cuda"
x, w = K.make_inputs(dev)
xf = x.cpu().numpy().astype(np.float64)
wf = w.cpu().numpy().astype(np.float64)

# float64 quantizer (same scheme as kernel/generator)
scale = np.max(np.abs(wf), axis=1, keepdims=True) / 7.0
q = np.clip(np.floor(wf / scale + 0.5), -7, 7) * scale
per_row_res = q - wf          # (m, k)
per_row_err = per_row_res @ xf  # (m,) exact per-row error of quantized matvec
err_norm = np.linalg.norm(per_row_err)
ref = wf @ xf
n = ref.size
den = max(np.linalg.norm(ref), 0.001*np.sqrt(n))
E_quant = err_norm / den

# actual kernel output vs its fp64 quant model and vs reference
out = K.run(x, w).cpu().numpy().astype(np.float64)
E_kernel = np.linalg.norm(out - ref) / den
E_kernel_vs_quantmodel = np.linalg.norm(out - (q @ xf)) / max(np.linalg.norm(q @ xf), 1e-12)

# directions in k-space
res_dir = per_row_res.sum(axis=0); res_dir /= np.linalg.norm(res_dir)
w_dir = wf.sum(axis=0); w_dir /= np.linalg.norm(w_dir)
# fraction of ||error||^2 explained by the aggregate residual direction:
# per-row error contribution of x's residual-aligned component
x_res_comp = float(np.dot(xf, res_dir))
x_w_comp = float(np.dot(xf, w_dir))
x_norm = float(np.linalg.norm(xf))
# benign random x baseline
rng = np.random.Generator(np.random.PCG64(1224))
xb = rng.standard_normal(256).astype(np.float64)
E_benign = np.linalg.norm(per_row_res @ xb) / max(np.linalg.norm(wf @ xb), 0.001*np.sqrt(n))
# residual-only amplification: error from residual-aligned part of x alone
x_only_res = xf - x_res_comp*0  # placeholder not needed
print(json.dumps({
  "E_quantmodel_fp64": float(E_quant),
  "E_kernel_actual": float(E_kernel),
  "E_kernel_vs_quantmodel": float(E_kernel_vs_quantmodel),
  "E_benign_random_x": float(E_benign),
  "err_norm": float(err_norm),
  "ref_norm": float(np.linalg.norm(ref)),
  "x_res_aligned_component": x_res_comp,
  "x_weight_aligned_component": x_w_comp,
  "x_norm": x_norm,
  "budget": 0.12,
  "E_over_budget": bool(E_kernel > 0.12),
  "adversarial_to_benign_ratio": float(E_kernel / E_benign)}))
