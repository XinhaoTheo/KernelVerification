import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_j")
import kernel as K

x, = K.make_inputs("cuda")
out = K.run(x)

xd = x.double().cpu().numpy()
mean = xd.sum() / 128
var = ((xd - mean) ** 2).sum() / 128
ref = (xd - mean) / np.sqrt(var + 1e-5)

out_np = out.float().cpu().numpy()
rel = np.linalg.norm(out_np - ref) / max(np.linalg.norm(ref), 0.001 * np.sqrt(128))

# also emulate kernel variance in fp32 for the c1 metric
xs = x.float().cpu().numpy().astype(np.float32)
total = np.float32(0.0); squares = np.float32(0.0)
for v in xs:
    total = np.float32(total + np.float32(v))
    squares = np.float32(squares + np.float32(v) * np.float32(v))
mean32 = np.float32(total / 128)
var32 = max(np.float32(squares / 128 - mean32 * mean32), np.float32(0.0))
var_rel_err = abs(float(var32) - float(var)) / float(var)

print(json.dumps({"metric": "relative L2 error of kernel output vs float64 reference",
    "reason": "contract tolerance is <=0.02 on relative L2 with this denominator",
    "rel_l2_error": float(rel), "tolerance": 0.02, "passes": bool(rel <= 0.02),
    "true_variance": float(var), "emulated_fp32_variance": float(var32),
    "variance_relative_error": var_rel_err, "output_finite": bool(np.isfinite(out_np).all()),
    "shape": list(out_np.shape), "dtype": str(out.dtype)}))