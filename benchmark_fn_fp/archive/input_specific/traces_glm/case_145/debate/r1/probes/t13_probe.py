
import json, math, numpy as np

rng = np.random.Generator(np.random.PCG64(203635))
low = rng.uniform(0.25, 1.75, (4, 8)).astype(np.float32)
interior = np.concatenate((np.full((4,1), 2.0**30, np.float32),
                           np.full((4,1), -(2.0**30), np.float32), low), axis=1)
order = rng.permutation(10)
x = np.empty((4,12), dtype=np.float32)
x[:,0] = np.float32(2.0**80); x[:,1:-1] = interior[:, order]; x[:,-1] = np.float32(-(2.0**80))

vals = x.astype(np.float64)
# replicate initial_probe reference: sequential float64 row-order accumulation
probe_ref = []
for r in range(4):
    acc = np.float64(0.0)
    for c in range(12):
        acc = np.float64(acc + vals[r,c])
    probe_ref.append(float(acc))
# exact real-number target
exact = [math.fsum(row) for row in vals]

print(json.dumps({
 "probe_reference_sequential_f64": probe_ref,
 "exact_target_fsum": exact,
 "mismatch": probe_ref != exact,
 "ulp_f64_at_2p80": 2.0**(80-52),
 "note": "initial_probe reference equals 0.0 while exact target is sum of eight small values"
}))
