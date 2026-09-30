import numpy as np, json, torch

rng = np.random.Generator(np.random.PCG64(830228))
u_np = rng.normal(size=32).astype(np.float32)
b_np = (1.125 * u_np.astype(np.float64) + 0.00001 * rng.normal(size=32)).astype(np.float32)

# actual kernel run on GPU
u = torch.from_numpy(u_np).to("cuda")
b = torch.from_numpy(b_np).to("cuda")
import triton
from triton_kernel_case_p import _projection_kernel  # placeholder; define below

# Instead, exec the artifact kernel source directly
import importlib.util
spec = importlib.util.spec_from_file_location("case_p_kernel", "/root/numerical_cases/case_p/kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
out = mod.run(u, b).cpu().numpy()

u64 = u_np.astype(np.float64); b64 = b_np.astype(np.float64)
alpha64 = np.dot(u64, b64) / np.dot(u64, u64)
res64 = b64 - alpha64 * u64
ref = res64 / np.linalg.norm(res64)

# isolate cancellation channel: fp32 residual computation using fp64 alpha
alpha64f = np.float64(alpha64)
res32 = np.zeros(32, dtype=np.float32)
for j in range(32):
    coef32 = np.float32(alpha64)  # quantize alpha to fp32 for the multiply
    res32[j] = np.float32(np.float32(b_np[j]) - np.float32(np.float32(u_np[j]) * coef32))
res32_64 = res32.astype(np.float64)
out_c = res32_64 / np.linalg.norm(res32_64)

kernel_err = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-12)
cancel_err = np.linalg.norm(out_c - ref) / max(np.linalg.norm(ref), 1e-12)

print(json.dumps({
    "kernel_output_finite": bool(np.all(np.isfinite(out))),
    "kernel_output_shape": list(out.shape),
    "kernel_rel_l2_error": float(kernel_err),
    "cancellation_channel_rel_l2_error": float(cancel_err),
    "tolerance": 0.01,
    "kernel_exceeds_tol": bool(kernel_err > 0.01),
    "cancellation_exceeds_tol": bool(cancel_err > 0.01),
    "median_abs_residual_true": float(np.median(np.abs(res64))),
}))