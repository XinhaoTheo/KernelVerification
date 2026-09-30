import json, sys
sys.path.insert(0, "/root/numerical_cases/case_e")
import numpy as np, torch
import kernel

matrix, initial, drive = kernel.make_inputs()
m = matrix.detach().cpu().numpy()  # float32
x0 = initial.detach().cpu().numpy().astype(np.float64)
d = drive.detach().cpu().numpy().astype(np.float64)
m64 = m.astype(np.float64)

# float64 reference
ref = x0.copy()
for t in range(64):
    ref = m64 @ ref + d[t]

# simulate kernel arithmetic: fp32 matvec + fp16 quantize each step
x = x0.astype(np.float32)
for t in range(64):
    nxt = (m.astype(np.float32) @ x.astype(np.float32)).astype(np.float32) + d[t].astype(np.float32)
    x = nxt.astype(np.float16).astype(np.float32)

# fp32-only simulation (no fp16 quantization) to isolate accumulation-order error
y = x0.astype(np.float32)
for t in range(64):
    y = (m.astype(np.float32) @ y.astype(np.float32)).astype(np.float32) + d[t].astype(np.float32)

def rel(v, ref):
    den = max(np.linalg.norm(ref), 0.001*np.sqrt(16))
    return float(np.linalg.norm(v.astype(np.float64) - ref) / den)

print(json.dumps({
    "sim_fp16_fp32_rel_l2": rel(x, ref),
    "sim_fp32only_rel_l2": rel(y, ref),
    "tolerance": 0.002,
    "sim_fp32only_passes": bool(rel(y, ref) <= 0.002),
    "fp16_contrib_dominant": bool(rel(x, ref) > 10 * max(rel(y, ref), 1e-12)),
    "note": "fp32-only sim isolates fp32 accumulation vs float64; difference between sims isolates fp16 quantization"
}))