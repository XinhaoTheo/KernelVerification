import numpy as np, torch, itertools, json
import sys
sys.path.insert(0, "/root/evidence_cases/case_e12")
from kernel import make_inputs_numpy, run, make_inputs

x, w, b, c = make_inputs_numpy()
W = w.astype(np.float64); B = b.astype(np.float64); C = c.astype(np.float64)
# per-neuron max over box: relu max = sum|W[j,:]| - b[j] (attained at x_i=sign(W[j,i]))
per_neuron = np.abs(W).sum(axis=1) - B  # = 0.25*sum|W[j]|
residual_max = (C * np.maximum(per_neuron, 0)).sum()
print("per_neuron_max_relu", per_neuron, "residual_upper_bound", residual_max)

# Exact per-neuron max is at sign(W[j]); find row maximizing residual by LP-free approach:
# residual is convex (sum of convex relu terms), max over box at a vertex. Enumerate smart: vertices 2^12=4096
best = -1; bestv = None
for bits in range(4096):
    v = np.array([1.0 if (bits>>k)&1 else -1.0 for k in range(12)])
    r = (C * np.maximum(W@v - B, 0)).sum()
    if r > best: best, bestv = r, v
print("vertex_max_residual", best)

# Also linear base error: kernel computes 0.25x0+0.5x1 exactly? compare. Error = residual itself.
X = torch.from_numpy(np.tile(bestv.astype(np.float32), (3,1))).cuda()
before = [t.clone() for t in make_inputs("cuda")]
out = run(X, *before[1:])
xd, Wd, Bd, Cd = [t.double() for t in make_inputs("cuda")]
for i in range(3):
    xd[i] = xd[i]
tgt = 0.25*xd[:,0]+0.5*xd[:,1] + torch.relu(xd @ Wd.T - Bd) @ Cd
err = (out.double() - tgt).abs().max().item()
print("kernel_max_abs_error_on_worst_vertex", err)
print(json.dumps({"residual_upper_bound": float(residual_max), "vertex_max_residual": float(best), "kernel_error_worst_vertex": err}))
