import sys, json
sys.path.insert(0, "/root/pilot_cases/case_95")
import torch, numpy as np
from kernel import make_inputs

x, w = make_inputs("cuda")
wf32 = w.cpu().numpy().astype(np.float32)
wf64 = wf32.astype(np.float64)
# fp32 path (mimics kernel): scale and q computed in fp32
s32 = np.max(np.abs(wf32), axis=1, keepdims=True).astype(np.float32) / np.float32(7.0)
q32 = np.clip(np.floor(wf32 / s32 + np.float32(0.5)), np.float32(-7.0), np.float32(7.0))
# fp64 path (generator's residual computation)
s64 = np.max(np.abs(wf64), axis=1, keepdims=True) / 7.0
q64 = np.clip(np.floor(wf64 / s64 + 0.5), -7, 7)
flips = int(np.count_nonzero(q32 != q64.astype(np.float32)))
# boundary proximity
dist = np.abs(wf64/s64 - np.round(wf64/s64) - 0.0)
frac = wf64/s64
near_half = int(np.count_nonzero(np.abs(frac - np.floor(frac) - 0.5) < 1e-5))
x64 = x.cpu().numpy().astype(np.float64)
err32 = float(np.linalg.norm((q32.astype(np.float64)*s32.astype(np.float64)) @ x64 - wf64 @ x64))
err64 = float(np.linalg.norm(q64*s64 @ x64 - wf64 @ x64))
print(json.dumps({"level_flips_fp32_vs_fp64": flips, "near_boundary_count": near_half,
 "err_norm_fp32_quant": err32, "err_norm_fp64_quant": err64,
 "total_elements": int(wf32.size)}))