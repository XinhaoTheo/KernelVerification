
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

targets = [sum(Fraction(float(v)) for v in r) for r in x]
exact = np.array([float(t) for t in targets])
exact_str = [str(t) for t in targets]

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

sim = np.array([sim_fp32(r) for r in x], dtype=np.float32).astype(np.float64)
err = float(np.linalg.norm(sim-exact)/max(np.linalg.norm(exact),1e-12))
rec = np.array([6.505321502685547, 7.741500377655029, 5.556467056274414, 7.19061279296875])
err_rec = float(np.linalg.norm(rec-exact)/max(np.linalg.norm(exact),1e-12))
print(json.dumps({"exact_targets": exact.tolist(), "exact_targets_fraction": exact_str,
  "sim_fp32": sim.tolist(), "sim_rel_err": err,
  "recorded_gpu_output": rec.tolist(), "recorded_gpu_rel_err_vs_exact": err_rec,
  "tolerance": 1e-5, "within_tolerance_sim": bool(err<=1e-5),
  "within_tolerance_recorded": bool(err_rec<=1e-5),
  "fp32_representable_gap": float(max(abs(np.float32(float(t))-float(t)) for t in targets))}))
