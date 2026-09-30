import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_i")
import kernel as K

x, = K.make_inputs()
out = K.run(x)

# float64 reference per problem.txt
xd = x.to(torch.float64).numpy()
mean = xd.sum() / 128
var = ((xd - mean) ** 2).sum() / 128
ref = (xd - mean) / np.sqrt(var + 1e-5)

err = np.linalg.norm(out.numpy().astype(np.float64) - ref) / max(np.linalg.norm(ref), 0.001 * np.sqrt(128))

# emulate kernel's float32 single-pass stats (sequential accumulation)
xf = x.numpy().astype(np.float32)
total = np.float32(0.0); squares = np.float32(0.0)
for v in xf:
    v = np.float32(v)
    total = np.float32(total + v)
    squares = np.float32(squares + np.float32(v * v))
mean_f = np.float32(total / np.float32(128))
sq_over_n = np.float32(squares / np.float32(128))
mean_sq = np.float32(mean_f * mean_f)
raw_var = np.float32(sq_over_n - mean_sq)
clamped = np.float32(max(raw_var, np.float32(0.0)))
denom = np.float32(np.sqrt(np.float32(clamped + np.float32(1e-5))))

print(json.dumps({
    "rel_l2_error": float(err),
    "tolerance": 0.02,
    "passes": bool(err <= 0.02),
    "out_finite": bool(np.isfinite(out.numpy()).all()),
    "float64_variance": float(var),
    "float32_single_pass_variance_raw": float(raw_var),
    "raw_negative": bool(float(raw_var) < 0),
    "clamped_to_zero": bool(float(clamped) == 0.0),
    "kernel_denominator_est": float(denom),
    "float64_denominator": float(np.sqrt(var + 1e-5)),
    "kernel_output_max_abs": float(np.abs(out.numpy()).max()),
}))
