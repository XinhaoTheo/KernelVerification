import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_i")
import kernel as K

x, = K.make_inputs()
out = K.run(x).cpu()

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

# recover kernel's effective denominator from output: out[i]*(denom) = x[i]-mean
# better: compare output scale against expected reference scale
xd = x.cpu().to(torch.float64).numpy()
mean = xd.sum() / 128
var = ((xd - mean) ** 2).sum() / 128
ref = (xd - mean) / np.sqrt(var + 1e-5)
scale = np.abs(out.numpy()).max() / np.abs(ref).max()

print(json.dumps({
    "float32_single_pass_variance_raw": float(raw_var),
    "raw_negative": bool(float(raw_var) < 0.0),
    "clamped_to_zero_effect": bool(float(raw_var) <= 0.0),
    "float64_variance": float(var),
    "kernel_output_max_abs": float(np.abs(out.numpy()).max()),
    "reference_output_max_abs": float(np.abs(ref).max()),
    "output_to_reference_max_ratio": float(scale),
    "out_finite": bool(np.isfinite(out.numpy()).all()),
}))
