import json, sys, math
sys.path.insert(0, "/root/pilot_cases/case_84")
import torch, numpy as np
import kernel as K

q, k, v = K.make_inputs("cuda")

# fp64 reference
q64 = q.double().cpu().numpy(); k64 = k.double().cpu().numpy(); v64 = v.double().cpu().numpy()
z = k64 @ q64 / math.sqrt(32); z -= z.max(); p = np.exp(z); p /= p.sum()
ref = p @ v64

# fp32-softmax-only simulation (no fp16 cast): replicate lines 12-14,16 in fp32
q32 = q.float().cpu().numpy(); k32 = k.float().cpu().numpy(); v32 = v.float().cpu().numpy()
s = (k32.astype(np.float64) @ q32) if False else None
# mimic Triton: dot in fp32 accumulation order ~ pairwise; use numpy float32 matmul with float64 accumulate not allowed -> use float32 GEMM
s32 = (k32 @ q32.astype(np.float32)) * (32 ** -0.5)
s32 = s32 - s32.max()
e = np.exp(s32.astype(np.float64)); e /= e.sum()
# keep softmax in fp32 precision
e32 = np.exp(s32); e32 = e32 / e32.sum()
y32 = (e32[:, None] * v32).sum(axis=0)  # fp32-ish accumulation

err = np.linalg.norm(y32.astype(np.float64) - ref)
refn = np.linalg.norm(ref)
denom = max(refn, 0.001*math.sqrt(32))
E2 = err/denom
# also with fp64 accumulate of the final weighted sum (isolating softmax from final sum)
y32b = (e32[:, None].astype(np.float64) * v64).sum(axis=0)
E2b = np.linalg.norm(y32b - ref)/denom
print(json.dumps({
    "metric": "E of fp32-softmax path (no fp16 cast) vs fp64 reference",
    "E_fp32_softmax": E2, "E_fp32_softmax_fp64sum": E2b,
    "ref_norm": refn, "denominator": denom, "finite": bool(np.isfinite(y32).all())
}))