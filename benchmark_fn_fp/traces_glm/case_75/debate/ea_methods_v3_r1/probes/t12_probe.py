import json, numpy as np, torch

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

# exact target in fp64
target = x.astype(np.float64).sum(axis=1)

# permutation position of the +2^30 slot (interior index 0)
c30_col = int(np.argmax(order == 0)) + 1  # column in 1..10
small_cols = [int(np.argmax(order == k)) + 1 for k in range(2, 10)]
small_before_c30 = sum(1 for c in small_cols if c < c30_col)

# emulate kernel fp32 fast-2sum chain (numpy fp32, no fusion)
def emulate(row):
    t = np.float32(0.0); corr = np.float32(0.0)
    for v in row:
        v = np.float32(v)
        u = np.float32(t + v)
        lost = np.float32((t - u) + v) if abs(t) >= abs(v) else np.float32((v - u) + t)
        corr = np.float32(corr + lost)
        t = u
    return np.float32(t + corr)
emulated = np.array([emulate(x[r]) for r in range(4)], dtype=np.float32)

# actual kernel
import kernel
xk, = kernel.make_inputs("cuda")
before = xk.clone()
out = kernel.run(xk)
out_np = out.detach().cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(out_np - target) / max(np.linalg.norm(target), 1e-12))
unmod = bool(torch.equal(xk, before))

print(json.dumps({
    "permutation_order": order.tolist(),
    "c30_column": c30_col,
    "small_value_columns": small_cols,
    "small_values_before_c30": small_before_c30,
    "exact_target_fp64": target.tolist(),
    "kernel_output": out_np.tolist(),
    "emulated_fp32_chain_output": emulated.astype(np.float64).tolist(),
    "relative_error_vs_exact": rel,
    "tolerance": 1e-5,
    "X_unmodified": unmod,
    "passes_contract": bool(rel <= 1e-5)
}))
