import importlib.util, json, math
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_i/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

(x,) = k.make_inputs("cuda")
out = k.run(x)
torch.cuda.synchronize()

xd = x.cpu().numpy().astype(np.float64)
mean64 = xd.sum() / 128
var64 = ((xd - mean64)**2).sum() / 128
ref = (xd - mean64) / math.sqrt(var64 + 1e-5)

od = out.cpu().numpy().astype(np.float64)
rel_l2 = np.linalg.norm(od - ref) / max(np.linalg.norm(ref), 0.001*math.sqrt(128))

# sequential float32 sum (kernel's order) and its mean error
tot = np.float32(0.0)
for v in x.cpu().numpy():
    tot = np.float32(tot + np.float32(v))
mean_f = float(tot)/128  # kernel divides float32 by 128 (exact power of 2)
mean_err = abs(mean_f - mean64)
dev_rms = float(np.sqrt((xd-mean64)**2).mean()**0.5) if False else float(np.sqrt(((xd-mean64)**2).mean()))

# mean-only counterfactual: use float32-sequential mean but float64 denominator
out_meanonly = (xd - mean_f) / math.sqrt(var64 + 1e-5)
rel_l2_meanonly = np.linalg.norm(out_meanonly - ref) / max(np.linalg.norm(ref), 0.001*math.sqrt(128))

print(json.dumps({
  "metric": "mean absolute error and mean-only rel-L2 counterfactual",
  "mean64": float(mean64), "mean_f32_seq": float(mean_f),
  "mean_abs_err": float(mean_err), "mean_rel_err_vs_dev": float(mean_err/0.125),
  "dev_rms": dev_rms,
  "rel_l2_full": float(rel_l2), "rel_l2_mean_only": float(rel_l2_meanonly),
  "tolerance": 0.02,
  "mean_only_breaches": bool(rel_l2_meanonly > 0.02),
  "full_breaches": bool(rel_l2 > 0.02)
}))