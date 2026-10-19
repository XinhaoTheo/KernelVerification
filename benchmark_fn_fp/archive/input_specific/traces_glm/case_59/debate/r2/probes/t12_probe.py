
import sys, json, numpy as np
sys.path.insert(0, "/root/numerical_cases/case_x")
import kernel

matrix_np = kernel.make_inputs_numpy()[0]
A64 = matrix_np.astype(np.float64)

# Reference: log(det) in float64 on stored float32 entries
sign, logdet_ref = np.linalg.slogdet(A64)

# Float32 replay of the kernel's exact no-pivot elimination (deterministic arithmetic, no fusion assumed: a-b*c with numpy is not fused)
M = matrix_np.astype(np.float32).copy()
pivots = []
for k in range(8):
    p = np.float32(M[k, k]); pivots.append(float(p))
    col = M[:, k].copy(); row = M[k, :].copy()
    mult = (col / p).astype(np.float32)
    updated = (M - np.outer(mult, row)).astype(np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[k+1:, k+1:] = True
    M = np.where(mask, updated, M).astype(np.float32)

# Actual kernel output
import torch
mat = torch.from_numpy(matrix_np).to("cuda")
out = kernel.run(mat)
kernel_out = float(out.cpu().numpy()[0])
finite = np.isfinite(kernel_out)

rel_err = abs(kernel_out - logdet_ref) / max(abs(logdet_ref), 1.0)
min_pivot = min(pivots)
neg_pivots = [i for i, p in enumerate(pivots) if p <= 0]

print(json.dumps({
    "pivots_float32": pivots,
    "min_pivot": min_pivot,
    "negative_pivots": neg_pivots,
    "kernel_output": kernel_out,
    "kernel_output_finite": bool(finite),
    "reference_logdet_float64": float(logdet_ref),
    "relative_error": float(rel_err),
    "tolerance": 1e-4,
    "claim_c1_confirmed": bool(min_pivot <= 1e-6 or (not finite) or rel_err > 1e-4)
}))
