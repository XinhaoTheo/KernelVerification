
import numpy as np, json, torch
rng = np.random.Generator(np.random.PCG64(782406))
x = (64.0 + rng.normal(0.0, 0.125, 128)).astype(np.float32)

# float64 two-pass reference
x64 = x.astype(np.float64)
mean64 = x64.sum()/128
var64 = ((x64-mean64)**2).sum()/128
ref = (x64-mean64)/np.sqrt(var64+1e-5)

# fp32 emulation of kernel accumulation order
total = np.float32(0.0); squares = np.float32(0.0)
for v in x:
    v = np.float32(v)
    total = np.float32(total + v)
    squares = np.float32(squares + np.float32(v*v))
mean32 = np.float32(total/np.float32(128))
sq_over_n = np.float32(squares/np.float32(128))
mean_sq = np.float32(mean32*mean32)
raw_sub = np.float32(sq_over_n - mean_sq)
var32 = np.float32(max(raw_sub, np.float32(0.0)))
denom32 = np.float32(np.sqrt(np.float32(var32+np.float32(1e-5))))

# actual GPU kernel
import sys
sys.path.insert(0, '/root/numerical_cases/case_j')
import kernel as K
xt = torch.from_numpy(x).to('cuda')
out = K.run(xt).cpu().numpy()

err = np.linalg.norm(out-ref)/max(np.linalg.norm(ref), 0.001*np.sqrt(128))
emul_err = np.linalg.norm(((x.astype(np.float64)-mean32)/denom32)-ref)/max(np.linalg.norm(ref),0.001*np.sqrt(128))
print(json.dumps({
  "relative_l2_error": float(err),
  "tolerance": 0.02,
  "raw_subtraction": float(raw_sub),
  "raw_subtraction_positive": bool(raw_sub > 0),
  "clamped": bool(raw_sub <= 0),
  "true_var": float(var64), "emul_var": float(var32),
  "ref_denom": float(np.sqrt(var64+1e-5)), "kernel_denom_emul": float(denom32),
  "denom_ratio": float(denom32/np.sqrt(var64+1e-5)),
  "emul_relative_l2": float(emul_err),
  "gpu_vs_emul_max_abs": float(np.abs(out-((x.astype(np.float32).astype(np.float64)-mean32)/denom32)).max()),
  "finite": bool(np.isfinite(out).all()), "shape": list(out.shape)
}))
