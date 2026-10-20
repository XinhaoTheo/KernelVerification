import numpy as np, torch, json, sys
sys.path.insert(0, "/root/evidence_cases/case_e12")
from kernel import make_inputs_numpy, run

smoke, w, b, c = make_inputs_numpy()
W = w.astype(np.float64); B = b.astype(np.float64); C = c.astype(np.float64)

best, bestv = -1, None
for bits in range(4096):
    v = np.array([1.0 if (bits>>k)&1 else -1.0 for k in range(12)])
    r = (C * np.maximum(W@v - B, 0)).sum()
    if r > best: best, bestv = r, v

Xn = np.tile(bestv.astype(np.float32), (4,1))
X = torch.from_numpy(Xn).cuda()
Wt, Bt, Ct = [torch.from_numpy(a).cuda() for a in (w, b, c)]
before = [t.clone() for t in (X, Wt, Bt, Ct)]
out = run(X, Wt, Bt, Ct)
Xd, Wd, Bd, Cd = [t.double() for t in (X, Wt, Bt, Ct)]
tgt = 0.25*Xd[:,0]+0.5*Xd[:,1] + torch.relu(Xd @ Wd.T - Bd) @ Cd
err = (out.double() - tgt).abs().max().item()
unmod = all(torch.equal(a, z) for a, z in zip((X, Wt, Bt, Ct), before))
print(json.dumps({
  "worst_vertex": bestv.tolist(),
  "vertex_max_residual": float(best),
  "kernel_max_abs_error": float(err),
  "tolerance": 1.0,
  "dtype": str(out.dtype), "shape": tuple(out.shape),
  "finite": bool(torch.isfinite(out).all()), "inputs_unmodified": bool(unmod),
  "violates_contract": bool(err > 1.0)
}))