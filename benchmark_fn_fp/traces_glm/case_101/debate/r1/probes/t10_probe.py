import sys, json
import numpy as np

# Exact emulation of the kernel's FP32 pipeline over a dense mantissa grid m in [1,2)
# no FMA (enable_fp_fusion=False), FP32 round-to-nearest ops, div_rn division.
def f32(x):
    return np.float32(x)

m = np.linspace(1.0, 2.0, 200001, dtype=np.float64).astype(np.float32)
true = np.cbrt(m.astype(np.float64))

def pipeline(mant, coeff):
    a, b, c = map(f32, coeff)
    est = f32(f32(a * mant) + b)
    est = f32(f32(est * mant) + c)
    est = f32(f32(f32(f32(2.0*est) + f32(mant/est)) ) * f32(1.0/3.0)) if False else est
    # Newton: (2*est + mant/(est*est)) * (1/3)  all fp32
    e2 = f32(est * est)
    div = f32(mant / e2)
    est = f32(f32(f32(2.0*est) + div) * f32(1.0/3.0))
    return est

def relerr(est):
    return np.max(np.abs(est.astype(np.float64) - true) / true)

trunc = (-0.0603, 0.4394, 0.6215)
est_trunc = pipeline(m, trunc)
rel_trunc = relerr(est_trunc)

# best-fit quadratic (numpy polyfit in double) as comparison
z = m.astype(np.float64)
p = np.polyfit(z, true, 2)
best = (p[0], p[1], p[2])
est_best = pipeline(m, best)
rel_best = relerr(est_best)

# seed-only error with truncated coefficients
seed = f32(f32(f32(f32(-0.0603*m) + f32(0.4394)) * m) + f32(0.6215))
seed_err = np.max(np.abs(seed.astype(np.float64) - true) / true)

res = {
  "metric": "max post-Newton relative error, exact FP32 pipeline emulation, dense grid m in [1,2), vs FP64 cbrt",
  "grid_points": int(m.size),
  "max_rel_err_truncated_coeffs": float(rel_trunc),
  "max_rel_err_bestfit_coeffs": float(rel_best),
  "best_fit_coeffs": [float(v) for v in best],
  "seed_only_max_rel_err_truncated": float(seed_err),
  "tolerance": 1e-6,
  "note": "post-Newton error with truncated 4-decimal coefficients vs a double-precision best-fit quadratic",
}
print(json.dumps(res))
