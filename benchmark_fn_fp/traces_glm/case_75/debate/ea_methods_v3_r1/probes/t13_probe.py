import json, numpy as np

SEED = 193602
rng = np.random.Generator(np.random.PCG64(SEED))
low = rng.uniform(0.25, 1.75, (4, 8)).astype(np.float32)
interior = np.concatenate((np.full((4,1), 2.0**30, np.float32),
                           np.full((4,1), -(2.0**30), np.float32), low), axis=1)
order = rng.permutation(10)
x = np.empty((4,12), dtype=np.float32)
x[:,0] = np.float32(2.0**80)
x[:,1:-1] = interior[:, order]
x[:,-1] = np.float32(-(2.0**80))

vals = x.astype(np.float64)
exact = vals.sum(axis=1)                      # mathematical exact sum
seq = np.zeros(4)
for c in range(12):                           # initial_probe reference: sequential fp64 in stored order
    seq = seq + vals[:, c]

print(json.dumps({
    "sequential_fp64_reference": seq.tolist(),
    "exact_target_fp64": exact.tolist(),
    "reference_equals_target": bool(np.array_equal(seq, exact)),
    "max_abs_difference": float(np.abs(seq - exact).max()),
    "small_value_sum_per_row": exact.tolist(),
    "conclusion": "probe reference (0.0) absorbs small values into 2^80; it is NOT the exact contract target" if not np.array_equal(seq, exact) else "probe reference equals target"
}))
