import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_j")
import kernel as K

x, = K.make_inputs("cuda")
out = K.run(x)

xs = x.float().cpu().numpy().astype(np.float32)
# emulate the kernel's sequential fp32 accumulation exactly
total = np.float32(0.0); squares = np.float32(0.0)
for v in xs:
    total = np.float32(total + np.float32(v))
    squares = np.float32(squares + np.float32(v) * np.float32(v))
mean32 = np.float32(total / 128)
diff = np.float32(np.float32(squares / 128) - np.float32(mean32 * mean32))

xd = x.double().cpu().numpy()
mean = xd.sum() / 128
var = ((xd - mean) ** 2).sum() / 128

# infer kernel variance from its actual output: out = (x-mean_k)/sqrt(v_k+1e-5)
# get denominator implied by output magnitude
den_k = float(np.abs(out.float().cpu().numpy()).max())  # not directly; instead use residual to reference
print(json.dumps({"metric": "sign of pre-clamp fp32 one-pass variance and implied kernel denominator",
    "reason": "c2 claims the fp32 difference goes negative and clamps variance to 0, making denominator ~0.0032",
    "pre_clamp_fp32_diff": float(diff), "is_negative": bool(float(diff) < 0.0),
    "true_variance": float(var),
    "implied_kernel_variance": float(np.float32(diff) ** 2 * 0 + max(float(diff), 0.0)),
    "denominator_if_clamped": float(np.sqrt(1e-5)),
    "true_denominator": float(np.sqrt(var + 1e-5))}))