import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e11/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

smoke, W, b, c = k.make_inputs_numpy()
Wd = W.astype(np.float64); bd = b.astype(np.float64); cd = c.astype(np.float64)

def residual(x):  # x: (n,12) float64
    return (np.maximum(x @ Wd.T - bd, 0.0) * cd).sum(axis=1)

# 1) all 2^12 box corners
signs = np.array(np.meshgrid(*([[-1,1]]*12))).T.reshape(-1,12).astype(np.float64)
corners = np.concatenate([signs, np.zeros((1,12),dtype=np.float64)])
res_c = residual(corners)
imax = int(np.argmax(res_c))
worst_corner = corners[imax]

# 2) per-row sign-aligned corners and center corner
center = np.zeros(12)
rng0 = np.random.Generator(np.random.PCG64(k.SEED))
center[:] = rng0.choice(np.asarray([-1,1]), size=12)  # not needed; derive center from W instead
# center = sign of any row's weights when flips absent is unknown; use W row signs:
row_aligned = np.sign(Wd)  # (6,12)
res_row = residual(row_aligned)

# 3) random corners for extra coverage
rng = np.random.Generator(np.random.PCG64(7))
rnd = rng.choice([-1.0,1.0], size=(4096,12))
res_rnd = residual(rnd)

max_res_corners = float(res_c.max())
max_res_rows = float(res_row.max())
max_res_rnd = float(res_rnd.max())

# 4) run actual kernel on the worst corner (legal X, contiguous, fp32, in [-1,1])
X = np.tile(worst_corner.astype(np.float32), (8,1))  # n=8
xt = torch.from_numpy(X).cuda(); wt = torch.from_numpy(W).cuda()
bt = torch.from_numpy(b).cuda(); ct = torch.from_numpy(c).cuda()
before = [v.clone() for v in (xt,wt,bt,ct)]
actual = k.run(xt, wt, bt, ct)
xd = xt.double(); wd=wt.double(); bd2=bt.double(); cd2=ct.double()
expected = 0.25*xd[:,0] + 0.5*xd[:,1] + (torch.relu(xd@wd.T - bd2) @ cd2)
err = (actual.double() - expected).abs()
struct_ok = (actual.dtype==torch.float32 and actual.shape==(8,) and bool(torch.isfinite(actual).all())
             and all(torch.equal(a,z) for a,z in zip((xt,wt,bt,ct), before)))

out = {
  "metric": "max omitted residual sum_j c[j]*relu(W[j].x-b[j]) over all 2^12 box corners + kernel abs error at worst corner, float64 reference",
  "reason": "contract requires EVERY x in [-1,1]^12 to have abs error <=1.0; residual is convex piecewise-linear so its box maximum is attained at a corner; kernel error == residual magnitude",
  "max_residual_all_corners": max_res_corners,
  "max_residual_per_row_sign_aligned": max_res_rows,
  "max_residual_random_4096_corners": max_res_rnd,
  "worst_corner": worst_corner.tolist(),
  "per_neuron_activation_at_worst": (np.maximum(worst_corner@Wd.T-bd,0)*cd).tolist(),
  "kernel_abs_error_at_worst_corner": float(err.max()),
  "tolerance": 1.0,
  "structural_ok": bool(struct_ok),
  "claim_breach_found": bool(max_res_corners > 1.0),
  "margin_to_tolerance": 1.0 - max_res_corners,
}
print(json.dumps(out))
