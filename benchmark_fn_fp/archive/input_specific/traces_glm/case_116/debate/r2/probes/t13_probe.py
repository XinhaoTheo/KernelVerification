
import sys, json, numpy as np
sys.path.insert(0, "/root/numerical_cases/case_x")
import kernel

matrix_np = kernel.make_inputs_numpy()[0]
A64 = matrix_np.astype(np.float64)
sign, logdet_ref = np.linalg.slogdet(A64)

# Float32 replay with per-step float32 log accumulation (kernel line 23)
M = matrix_np.astype(np.float32).copy()
acc = np.float32(0.0)
pivots_f32, logs_f32 = [], []
for k in range(8):
    p = np.float32(M[k, k]); pivots_f32.append(float(p))
    lp = np.float32(np.log(p)); logs_f32.append(float(lp))
    acc = np.float32(acc + lp)
    col = M[:, k].copy(); row = M[k, :].copy()
    mult = (col / p).astype(np.float32)
    updated = (M - np.outer(mult, row)).astype(np.float32)
    mask = np.zeros((8, 8), dtype=bool); mask[k+1:, k+1:] = True
    M = np.where(mask, updated, M).astype(np.float32)

# Float64 replay of the same no-pivot elimination (algorithm error only)
M64 = A64.copy()
piv64 = []
for k in range(8):
    piv64.append(M64[k, k])
    mult = M64[:, k] / M64[k, k]
    M64 = M64 - np.outer(mult, M64[k, :])
logdet_replay64 = float(np.sum(np.log(piv64)))

import torch
mat = torch.from_numpy(matrix_np).to("cuda")
out = kernel.run(mat)
kernel_out = float(out.cpu().numpy()[0])

def rel(x): return abs(x - logdet_ref) / max(abs(logdet_ref), 1.0)

print(json.dumps({
    "all_pivots_positive": bool(min(pivots_f32) > 0),
    "pivots_float32": pivots_f32,
    "log_terms_float32": logs_f32,
    "replay_f32_logdet": float(acc),
    "replay_f64_logdet": logdet_replay64,
    "reference_float64": float(logdet_ref),
    "kernel_output": kernel_out,
    "rel_err_kernel_vs_ref": float(rel(kernel_out)),
    "rel_err_replay32_vs_ref": float(rel(float(acc))),
    "rel_err_replay64_vs_ref": float(rel(logdet_replay64)),
    "log_accum_only_error": float(abs(float(acc) - logdet_replay64)),
    "tolerance": 1e-4,
    "claim_c2_confirmed": bool(min(pivots_f32) > 0 and rel(kernel_out) > 1e-4)
}))
