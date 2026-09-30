
import numpy as np, json
from fractions import Fraction

SEED = 193662
rng = np.random.Generator(np.random.PCG64(SEED))
low = rng.uniform(0.25, 1.75, (4, 8)).astype(np.float32)
interior = np.concatenate((np.full((4,1), 2.0**30, np.float32),
                           np.full((4,1), -(2.0**30), np.float32), low), axis=1)
order = rng.permutation(10)
x = np.empty((4,12), dtype=np.float32)
x[:,0] = np.float32(2.0**80)
x[:,1:-1] = interior[:, order]
x[:,-1] = np.float32(-(2.0**80))

pos = np.where(order==0)[0][0]; neg = np.where(order==1)[0][0]
lo, hi = min(pos,neg), max(pos,neg)
lows_between = [int(i) for i in range(lo+1, hi) if i not in (pos,neg)]

def sim_fp32(row):
    t = np.float32(0.0); c = np.float32(0.0)
    for v in row:
        v = np.float32(v)
        u = np.float32(t + v)
        if abs(t) >= abs(v): lost = np.float32(np.float32(t-u)+v)
        else: lost = np.float32(np.float32(v-u)+t)
        c = np.float32(c + lost)
        t = u
    return np.float32(t + c)

sim = np.array([sim_fp32(r) for r in x], dtype=np.float32)
targets = [sum(Fraction(v) for v in r) for r in x]
exact = np.array([float(t) for t in targets])
err = float(np.linalg.norm(sim.astype(np.float64)-exact)/max(np.linalg.norm(exact),1e-12))
print(json.dumps({"perm_order": order.tolist(), "pos30_idx": int(pos), "neg30_idx": int(neg),
  "lows_between_pm2p30_count": len(lows_between), "lows_between": lows_between,
  "exact_targets": exact.tolist(), "sim_fp32_outputs": sim.astype(np.float64).tolist(),
  "relative_error_vs_exact": err, "tolerance": 1e-5,
  "absorption_fires": bool(len(lows_between)>0), "max_row_abs_error": float(np.max(np.abs(sim.astype(np.float64)-exact)))}))
