import numpy as np, json

rng = np.random.Generator(np.random.PCG64(830228))
u = rng.normal(size=32).astype(np.float32)
b = (1.125 * u.astype(np.float64) + 0.00001 * rng.normal(size=32)).astype(np.float32)

u64 = u.astype(np.float64); b64 = b.astype(np.float64)
alpha64 = np.dot(u64, b64) / np.dot(u64, u64)

# emulate kernel's sequential fp32 accumulation (separately rounded products/sums)
num = np.float32(0.0); den = np.float32(0.0)
for j in range(32):
    num = np.float32(num + np.float32(np.float32(u[j]) * np.float32(b[j])))
    den = np.float32(den + np.float32(np.float32(u[j]) * np.float32(u[j])))
alpha32 = np.float32(np.divide(num, den, dtype=np.float32))

d_alpha = float(alpha64) - float(alpha32)
# reference: float64 residual (no cancellation issue in fp64 here, but recenter via fp32-quantized exact subtraction is fine)
res64 = b64 - alpha64 * u64
ref = res64 / np.linalg.norm(res64)

# isolate alpha channel: float64 residual using fp32 alpha
res_a = b64 - float(alpha32) * u64
out_a = res_a / np.linalg.norm(res_a)
err_a = np.linalg.norm(out_a - ref) / max(np.linalg.norm(ref), 1e-12)

print(json.dumps({
    "alpha64": float(alpha64), "alpha32": float(alpha32),
    "delta_alpha": d_alpha, "rel_delta_alpha": d_alpha / alpha64,
    "alpha_channel_rel_l2_error": float(err_a),
    "tolerance": 0.01,
    "alpha_channel_exceeds_tol": bool(err_a > 0.01),
    "median_abs_residual": float(np.median(np.abs(res64))),
    "delta_alpha_u_rms": float(np.sqrt(np.mean((d_alpha * u64) ** 2))),
}))