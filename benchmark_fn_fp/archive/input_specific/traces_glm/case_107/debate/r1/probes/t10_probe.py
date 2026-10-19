import json, numpy as np, torch, triton
import sys
sys.path.insert(0, "/root/numerical_cases/case_o")
import kernel as K

u, b = K.make_inputs("cuda")
out = K.run(u, b)
torch.cuda.synchronize()
out_np = out.detach().cpu().numpy().astype(np.float64)

u64 = u.detach().cpu().numpy().astype(np.float64)
b64 = b.detach().cpu().numpy().astype(np.float64)

# fp64 reference (recentred-free, cancellation negligible in fp64)
alpha = (u64 * b64).sum() / (u64 * u64).sum()
res = b64 - alpha * u64
ref = res / np.linalg.norm(res)

err = np.linalg.norm(out_np - ref) / max(np.linalg.norm(ref), 1e-12)

# fp32 CPU emulation of kernel arithmetic
u32 = u.detach().cpu().numpy(); b32 = b.detach().cpu().numpy()
num = np.float32(0.0); den = np.float32(0.0)
for j in range(32):
    num = np.float32(num + np.float32(u32[j] * b32[j]))
    den = np.float32(den + np.float32(u32[j] * u32[j]))
coef = np.float32(num / den)
nsq = np.float32(0.0)
res32 = np.empty(32, dtype=np.float32)
for j in range(32):
    res32[j] = np.float32(b32[j] - np.float32(u32[j] * coef))
    nsq = np.float32(nsq + np.float32(res32[j] * res32[j]))
norm = np.float32(np.sqrt(nsq))
emu = (res32 / norm).astype(np.float64)
err_emu = np.linalg.norm(emu - ref) / max(np.linalg.norm(ref), 1e-12)
bitmatch = bool(np.array_equal(emu.astype(np.float32), out.detach().cpu().numpy()))
res_norm = float(np.linalg.norm(res))
finite = bool(np.all(np.isfinite(out_np)))

print(json.dumps({
  "rel_l2_gpu_vs_fp64ref": float(err),
  "rel_l2_fp32emu_vs_fp64ref": float(err_emu),
  "bitmatch_gpu_vs_emu": bitmatch,
  "alpha_fp64": float(alpha), "alpha_fp32_kernel": float(coef),
  "residual_norm_fp64": res_norm,
  "tolerance": 0.01,
  "passes_tolerance": bool(err <= 0.01),
  "output_finite": finite,
  "shape": list(out_np.shape)
}))