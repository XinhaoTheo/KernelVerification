import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_i")
import kernel as K

x, = K.make_inputs()
out = K.run(x).cpu()

xd = x.cpu().to(torch.float64).numpy()
mean = xd.sum() / 128
var = ((xd - mean) ** 2).sum() / 128
ref = (xd - mean) / np.sqrt(var + 1e-5)

err = np.linalg.norm(out.numpy().astype(np.float64) - ref) / max(np.linalg.norm(ref), 0.001 * np.sqrt(128))

# emulate kernel's float32 single-pass stats (sequential accumulation)
xf = x.cpu().numpy().astype(np.float32)
total = np.float32(0.0); squares = np.float32(0.0)
for v in xf:
    v = np.float32(v)
    total = np.float32(total + v)
    squares = np.float32(squares + np.float32(v * v))
mean_f = np.float32(total / np.float32(128))
sq_over_n = np.float32(squares / np.float32(128))
mean_sq = np.float32(mean_f * mean_f)
raw_var = np.float32(sq_over_n - mean_sq)

print(json.dumps({
    "rel_l2_error": float(err),
    "tolerance": 0.02,
    "passes": bool(err <= 0.02),
    "out_finite": bool(np.isfinite(out.numpy()).all()),
    "float64_variance": float(var),
    "float32_single_pass_variance_raw": float(raw_var),
    "variance_rel_error": float(abs(float(raw_var) - var) / var),
    "float64_denominator": float(np.sqrt(var + 1e-5)),
    "kernel_output_max_abs": float(np.abs(out.numpy()).max()),
}))
