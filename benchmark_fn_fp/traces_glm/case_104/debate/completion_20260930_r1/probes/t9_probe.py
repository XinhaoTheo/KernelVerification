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
per_row_res = q - wf  # per-row quantization residual vectors
quant_out = (q @ xf)  # what the kernel approximates (in fp32, ~this)
ref = wf @ xf
err = quant_out - ref
res_dir = per_row_res.sum(axis=0); res_dir /= np.linalg.norm(res_dir)
w_dir = wf.sum(axis=0); w_dir /= np.linalg.norm(w_dir)
# residual-aligned component of error
err_aligned = float(np.dot(err, res_dir))
n = ref.size
E = np.linalg.norm(err) / max(np.linalg.norm(ref), 0.001*np.sqrt(n))
# actual kernel output for comparison
out = K.run(x, w).cpu().numpy().astype(np.float64)
E_kernel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(n))
# benign random x baseline
rng = np.random.Generator(np.random.PCG64(1224))
xb = rng.standard_normal(256).astype(np.float64)
E_benign = np.linalg.norm(q @ xb - wf @ xb) / max(np.linalg.norm(wf @ xb), 0.001*np.sqrt(n))
print(json.dumps({"E_quantmodel_fp64": float(E), "E_kernel_actual": float(E_kernel),
                  "err_norm": float(np.linalg.norm(err)),
                  "err_residual_aligned_component": err_aligned,
                  "cos_err_residual_dir": float(np.dot(err/np.linalg.norm(err), res_dir)),
                  "x_align_weight_dir": float(np.dot(xf/np.linalg.norm(xf), w_dir)),
                  "E_benign_random_x": float(E_benign),
                  "budget": 0.12}))
