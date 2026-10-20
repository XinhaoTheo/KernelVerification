import sys, json
sys.path.insert(0, "/root/numerical_cases/case_z")
import numpy as np, torch, kernel

q, a, v = kernel.make_inputs()
out = kernel.run(q, a, v)

# float64 reference from stored float32 values
q64 = q.cpu().numpy().astype(np.float64)
a64 = a.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
d_true = ((a64 - q64) ** 2).sum(axis=1)
w_true = np.exp(-16.0 * d_true)
ref = (w_true * v64).sum() / w_true.sum()

# emulate kernel's fp32 expanded-distance arithmetic exactly
def fp32(x): return np.float32(x)
d_comp = np.zeros(16, dtype=np.float32)
for i in range(16):
    an = np.float32(0.0); qn = np.float32(0.0); dot = np.float32(0.0)
    for k in range(32):
        aa = fp32(a64[i, k]); qq = fp32(q64[k])
        an = fp32(an + fp32(aa * aa))
        qn = fp32(qn + fp32(qq * qq))
        dot = fp32(dot + fp32(aa * qq))
    ns = fp32(an + qn); td = fp32(2.0 * dot)
    raw = np.float32(ns - td)
    d_comp[i] = max(raw, np.float32(0.0))

kernel_out = float(out.item())
err = abs(kernel_out - ref) / max(abs(ref), 0.05)
num_clamped = int(((a64**2).sum(1) + (q64**2).sum() - 2*(a64*q64).sum(1) < 0).sum())
print(json.dumps({
    "kernel_output": kernel_out,
    "reference": float(ref),
    "contract_error": float(err),
    "tolerance": 0.05,
    "passes_contract": bool(err <= 0.05),
    "true_distances": [float(x) for x in d_true],
    "computed_distances_emulated": [float(x) for x in d_comp],
    "max_distance_abs_err": float(np.max(np.abs(d_comp - d_true))),
    "negative_raw_count": num_clamped,
    "all_computed_zero": bool(np.all(d_comp == 0)),
    "finite_output": bool(np.isfinite(kernel_out)),
}))
