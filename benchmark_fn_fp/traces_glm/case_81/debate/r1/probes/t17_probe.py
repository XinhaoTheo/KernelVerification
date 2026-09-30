import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e20/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, = k.make_inputs()
before = x.clone()
out = k.run(x)

# Exact real-number target: +2^80/-2^80 and +2^30/-2^30 pairs are identical stored
# values and cancel exactly in real arithmetic, so target[r] = sum of the 8 low values
# (stored fp32, exactly representable in float64; summing 8 of them in f64 is exact).
SEED = 203604
rng = np.random.Generator(np.random.PCG64(SEED))
low = rng.uniform(0.25, 1.75, (4, 8)).astype(np.float32)
_ = rng.permutation(10)  # consume same RNG state as make_inputs_numpy
target = low.astype(np.float64).sum(axis=1)

# Cross-check: lows extracted from the actual tensor x match the RNG lows
xs = before.detach().cpu().numpy()
mask = (np.abs(xs) < 2)
lows_from_x = [np.sort(xs[r][mask[r]].astype(np.float64)) for r in range(4)]
lows_from_rng = [np.sort(low[r].astype(np.float64)) for r in range(4)]
lows_match = all(np.array_equal(a, b) for a, b in zip(lows_from_x, lows_from_rng))

# Independent exact check with Python fractions/decimal-free: sum via math.fsum on full row
import math
fsum_target = [math.fsum(xs[r].astype(np.float64).tolist()) for r in range(4)]

actual = out.detach().cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(actual - target) / max(np.linalg.norm(target), 1e-12))
print(json.dumps({
  "output": actual.tolist(),
  "exact_target_sum_of_lows": target.tolist(),
  "fsum_full_row_target": fsum_target,
  "lows_from_x_match_rng": bool(lows_match),
  "relative_error": rel,
  "tolerance": 1e-5,
  "input_unmodified": bool(torch.equal(x, before)),
  "finite": bool(np.isfinite(actual).all()),
  "passes_contract": bool(rel <= 1e-5)
}))