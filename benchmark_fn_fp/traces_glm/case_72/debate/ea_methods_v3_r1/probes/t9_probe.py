import importlib.util, json, itertools
import numpy as np, torch

spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

smoke, W, b, c = k.make_inputs_numpy()
Wd, bd, cd = [torch.from_numpy(a).double() for a in (W, b, c)]

# All 4096 sign corners of [-1,1]^12; residual = sum_j 0.25*max(W[j].x - b[j],0)
# is convex (sum of convex), so its box maximum is attained at a vertex.
corners = torch.from_numpy(np.array(list(itertools.product([-1.0, 1.0], repeat=12)), dtype=np.float64))
# chunk to keep memory sane
best, best_x = -1.0, None
for i in range(0, 4096, 1024):
    r = torch.relu(corners[i:i+1024] @ Wd.T - bd) @ cd
    m = int(r.argmax()); 
    if float(r[m]) > best:
        best = float(r[m]); best_x = corners[i+m]
# per-neuron max pre-activation over box = |W[j]|_1 (at x=sign(W[j]))
perneuron = (np.abs(W).sum(axis=1) - b)  # theoretical max (W[j].x - b[j]) at aligned corner
print(json.dumps({
  "metric": "max over box of dropped residual sum_j c[j]*max(W[j].x-b[j],0), float64, evaluated at all 4096 corners (convex -> vertex max)",
  "max_dropped_residual_box": best,
  "tolerance": 1.0,
  "best_x": [float(v) for v in best_x],
  "theoretical_bound_0.0625_sum_absW": 0.0625*float(np.abs(W).sum()),
  "sum_abs_W": float(np.abs(W).sum()),
  "per_neuron_max_excess_theoretical": [round(float(v), 4) for v in perneuron],
  "max_residual_exceeds_1.0": bool(best > 1.0),
}))