import numpy as np, json
SEED = 203604
rng = np.random.Generator(np.random.PCG64(SEED))
low = rng.uniform(0.25, 1.75, (4, 8)).astype(np.float32)
interior = np.concatenate((np.full((4,1), 2.0**30, np.float32),
                           np.full((4,1), -(2.0**30), np.float32), low), axis=1)
order = rng.permutation(10)
x = np.empty((4,12), dtype=np.float32)
x[:,0] = np.float32(2.0**80)
x[:,1:-1] = interior[:, order]
x[:,-1] = np.float32(-(2.0**80))

# Emulate kernel's fp32 Neumaier accumulation exactly (np.float32 ops, no fusion)
f = np.float32
results = []
trace = []
for r in range(4):
    total = f(0.0); corr = f(0.0)
    for c in range(12):
        v = x[r, c]
        upd = f(total + v)
        if abs(total) >= abs(v):
            lost = f(f(total - upd) + v)
        else:
            lost = f(f(v - upd) + total)
        corr = f(corr + lost)
        total = upd
    results.append(float(f(total + corr)))
    trace.append(float(corr))

# permutation positions: where do +2^30 (interior col 0) and -2^30 (col 1) land
p230 = int(np.where(order == 0)[0][0]); m230 = int(np.where(order == 1)[0][0])
target = x.astype(np.float64).sum(axis=1)
print(json.dumps({
  "permutation_order": order.tolist(),
  "pos_plus_2p30_in_interior": p230, "pos_minus_2p30_in_interior": m230,
  "num_smalls_between_plus_and_minus_2p30": abs(p230 - m230) - 1,
  "emulated_final_correction": trace,
  "emulated_output": results,
  "exact_target": target.tolist(),
  "fp32_ulp_at_2p30": 128.0
}))