import json, numpy as np, torch
import sys
sys.path.insert(0, '/root/numerical_cases/case_u')
from kernel import make_inputs, run, make_inputs_numpy

coef, offset = make_inputs()
out = run(coef, offset).cpu().numpy()

c64 = make_inputs_numpy()[0].astype(np.float64)
off64 = float(make_inputs_numpy()[1][0])
n64 = np.arange(64, dtype=np.float64)

# fp64 reference for RETAINED modes only (k=1..6) -- isolates fp32 angle/accum error
ref_ret = off64 * np.ones(64)
for k in range(1, 7):
    ref_ret += c64[0, k-1] * np.cos(2*np.pi*k*n64/64) + c64[1, k-1] * np.sin(2*np.pi*k*n64/64)

err = np.linalg.norm(out.astype(np.float64) - ref_ret)
ref_full_norm = np.linalg.norm([off64]) # not needed; use ref_ret norm
ratio_ret = err / max(np.linalg.norm(ref_ret), 1e-12)
max_abs = float(np.max(np.abs(out - ref_ret)))

# fp64 all-mode reference ratio for context
ref_all = off64 * np.ones(64)
for k in range(1, 17):
    ref_all += c64[0, k-1] * np.cos(2*np.pi*k*n64/64) + c64[1, k-1] * np.sin(2*np.pi*k*n64/64)
ratio_all = np.linalg.norm(out - ref_all) / max(np.linalg.norm(ref_all), 1e-12)

print(json.dumps({
    "metric": "relative L2 of kernel output vs fp64 retained-modes-only reference (isolates fp32 angle/accumulation error)",
    "ratio_retained_modes_fp32_vs_fp64": float(ratio_ret),
    "max_abs_err_retained": max_abs,
    "ratio_vs_full_reference": float(ratio_all),
    "tolerance": 0.15,
    "fp32_error_below_tolerance": bool(ratio_ret < 1e-3),
}))
