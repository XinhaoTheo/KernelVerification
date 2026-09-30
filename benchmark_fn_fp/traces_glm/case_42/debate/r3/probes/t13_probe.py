import sys, json
sys.path.insert(0, "/root/numerical_cases/case_g")
import torch, numpy as np
from kernel import make_inputs_numpy

vals, = make_inputs_numpy()  # (64,128) float32, contiguous
ref = vals.astype(np.float64).sum(axis=1)

total_small, absorbed = 0, 0
max_abs_partial = 0.0
outs = []
absorbed_mask = np.zeros(vals.shape, dtype=bool)
for i in range(64):
    acc = np.float32(0.0)
    partials = []
    for j in range(128):
        v = vals[i, j]
        new = np.float32(acc + v)
        is_small = abs(v) <= 1.0
        if is_small:
            total_small += 1
            if new == acc:
                absorbed += 1
                absorbed_mask[i, j] = True
        acc = new
        partials.append(float(acc))
    outs.append(float(acc))
    max_abs_partial = max(max_abs_partial, max(abs(p) for p in partials))

outs = np.array(outs, dtype=np.float64)
err = outs - ref
E = float(np.linalg.norm(err) / max(np.linalg.norm(ref), 0.008))
print(json.dumps({
    "total_small_entries": total_small,
    "absorbed_small_entries": absorbed,
    "absorbed_fraction": absorbed / total_small,
    "max_abs_partial_sum": max_abs_partial,
    "sim_E": E,
    "sim_max_abs_row_err": float(np.abs(err).max()),
    "sim_matches_kernel_expectation_note": "ascending-order fp32 sequential accumulation, same as kernel with enable_fp_fusion=False",
}))