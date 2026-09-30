import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_k")
import kernel

A32, b32 = kernel.make_inputs_numpy()
A = A32.astype(np.float64); b = b32.astype(np.float64)
x = np.linalg.solve(A, b)

# fp64 64-step Richardson (pure truncation)
s64 = np.zeros(16)
for _ in range(64):
    s64 = s64 + (b - A @ s64)
# fp32 64-step Richardson (truncation + rounding, mimicking kernel ops)
s32 = np.zeros(16, dtype=np.float32)
A32m, b32v = A32, b32
for _ in range(64):
    prod = (A32m * s32[None, :]).astype(np.float32)
    acc = prod.sum(axis=1, dtype=np.float32)  # fp32 accumulation
    resid = (b32v - acc).astype(np.float32)
    s32 = (s32 + resid).astype(np.float32)

den = max(np.linalg.norm(x), 0.001*np.sqrt(16))
e64 = np.linalg.norm(s64 - x) / den
e32 = np.linalg.norm(s32.astype(np.float64) - x) / den
print(json.dumps({
    "metric": "relerr of fp32 vs fp64 64-step Richardson simulation vs exact fp64 solve",
    "reason": "fp64 loop isolates truncation; fp32 loop adds rounding; difference is the rounding contribution vs the 0.08 bound",
    "relerr_fp64_iteration": float(e64),
    "relerr_fp32_iteration": float(e32),
    "rounding_increment": float(e32 - e64),
    "fp32_alone_passes": bool(e32 <= 0.08),
    "fp64_iteration_passes": bool(e64 <= 0.08),
    "max_abs_diff_fp32_vs_fp64_states": float(np.max(np.abs(s32.astype(np.float64) - s64))),
    "tolerance": 0.08
}))
