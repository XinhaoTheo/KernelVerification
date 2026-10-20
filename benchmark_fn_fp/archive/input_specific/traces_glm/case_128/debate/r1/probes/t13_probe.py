import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/numerical_cases/case_x/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

(matrix,) = kern.make_inputs()
out = kern.run(matrix)
kernel_val = float(out.item())

A64 = matrix.cpu().numpy().astype(np.float64)
ref = float(np.log(np.linalg.det(A64)))

# fp32 emulation with exact float64 log of fp32 pivots -> isolates log approximation
M = matrix.cpu().numpy().astype(np.float32).copy()
pivots = []
for k in range(8):
    pivots.append(np.float32(M[k, k]))
    mult = (M[:, k] / pivots[-1]).astype(np.float32)
    prow = M[k, :].astype(np.float32)
    updated = (M - np.multiply.outer(mult, prow).astype(np.float32)).astype(np.float32)
    mask = np.zeros((8,8), bool); mask[k+1:, k+1:] = True
    M = np.where(mask, updated, M)
pivots = np.array(pivots, np.float32)
logsum_exact_log = float(np.sum([np.log(np.float64(p)) for p in pivots]))

err_kernel = abs(kernel_val - ref) / max(abs(ref), 1.0)
err_exactlog = abs(logsum_exact_log - ref) / max(abs(ref), 1.0)
print(json.dumps({
    "kernel_output": kernel_val,
    "reference_logdet_f64": ref,
    "kernel_relative_error": err_kernel,
    "fp32_elim_exactlog_sum": logsum_exact_log,
    "fp32_elim_exactlog_relative_error": err_exactlog,
    "log_approx_error_contribution": abs(kernel_val - logsum_exact_log),
    "pivots": [float(p) for p in pivots],
    "max_abs_log_pivot": float(max(abs(np.log(np.float64(p))) for p in pivots)),
    "tolerance": 1e-4,
    "kernel_exceeds_tolerance": bool(err_kernel > 1e-4),
}))
