import numpy as np

rng = np.random.Generator(np.random.PCG64(98262))
factor = rng.integers(-8, 9, (8, 7), dtype=np.int64)
gram = np.sum(factor[:, None, :] * factor[None, :, :], axis=2, dtype=np.int64)
A = (gram.astype(np.float64) + (1.0/1024.0)*np.eye(8)).astype(np.float32)

ref = np.float64(np.log(np.linalg.det(A.astype(np.float64))))

# Emulate kernel's FP32 elimination without pivoting
M = A.astype(np.float32).copy()
pivots = []
for k in range(8):
    pivot = np.float32(M[k, k])
    pivots.append(pivot)
    col = M[:, k].astype(np.float32)
    prow = M[k, :].astype(np.float32)
    mult = (col / pivot).astype(np.float32)  # div_rn fp32
    product = np.multiply.outer(mult, prow).astype(np.float32)
    updated = (M - product).astype(np.float32)
    mask = np.zeros((8,8), dtype=bool)
    mask[k+1:, k+1:] = True
    M = np.where(mask, updated, M)
pivots = np.array(pivots, dtype=np.float32)
logsum = np.float32(0.0)
for p in pivots:
    logsum = np.float32(logsum + np.float32(np.log(p)))  # fp32 accumulate, exact log approx
err = abs(float(logsum) - float(ref)) / max(abs(float(ref)), 1.0)
import json
print(json.dumps({
    "reference_logdet_f64": float(ref),
    "emulated_fp32_logsum": float(logsum),
    "relative_error": err,
    "tolerance": 1e-4,
    "pivots_fp32": [float(p) for p in pivots],
    "min_pivot": float(pivots.min()),
    "all_pivots_positive": bool((pivots > 0).all()),
    "exceeds_tolerance": bool(err > 1e-4),
}))
