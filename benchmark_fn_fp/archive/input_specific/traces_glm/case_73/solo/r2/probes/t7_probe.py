import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e12/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

smoke, W, b, c = k.make_inputs_numpy()
Wd, bd, cd = [torch.from_numpy(v).double() for v in (W, b, c)]
# per-neuron max of relu(dot(W_j,x)-b_j) over x in [-1,1]^12 is attained at x = sign(W_j) (all entries +-1): dot = |W_j| sum
relu_max = (Wd.abs().sum(1) - bd).clamp(min=0)
residual_max = float((cd * relu_max).sum())
print("per-neuron relu max:", relu_max.tolist(), "|W| sums:", Wd.abs().sum(1).tolist(), "b:", bd.tolist())
print("worst-case dropped residual:", residual_max)

# adversarial inputs: x = sign(W_j) for each row j, plus corners of the box
adv = [np.sign(W[j]).astype(np.float32) for j in range(6)]
rng = np.random.default_rng(0)
for _ in range(2000):
    adv.append(rng.choice([-1.0,1.0], size=12).astype(np.float32))
X = torch.from_numpy(np.stack(adv)).cuda()
Xf, Wt, Bt, Ct = [torch.from_numpy(v).cuda() for v in (X.cpu().numpy().astype(np.float32), W, b, c)]
out = k.run(Xf, Wt, Bt, Ct)
Xd, Wd2, Bd2, Cd2 = [v.double() for v in (Xf, Wt, Bt, Ct)]
ref = 0.25*Xd[:,0] + 0.5*Xd[:,1] + torch.relu(Xd @ Wd2.T - Bd2) @ Cd2
err = (out.double() - ref).abs()
best_row = Xf[err.argmax()].cpu().numpy()
res = {
  "metric": "max absolute error vs float64 full target (linear base + ReLU residual)",
  "n_adversarial_rows": X.shape[0],
  "max_absolute_error": float(err.max()),
  "tolerance": 1.0,
  "worst_row": best_row.tolist(),
  "analytic_worst_residual_bound": residual_max,
  "violation_found": bool(err.max() > 1.0),
  "output_finite_shape_dtype_ok": bool(out.dtype==torch.float32 and out.shape==(X.shape[0],) and torch.isfinite(out).all()),
}
print(json.dumps(res, indent=2))
