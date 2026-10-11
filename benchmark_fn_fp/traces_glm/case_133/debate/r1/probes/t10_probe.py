import torch, json, sys, math
sys.path.insert(0, "/root/cases/case_133")
import kernel

dev = "cuda"
# find FP16 values where x*(2/pi)+0.5 is very close to an integer (near quadrant midpoints of reduction)
# quadrant boundaries at x ~ k*pi/2 + pi/4-ish; instead scan all FP16 values in a range around typical magnitude
# Brute force scan of all FP16 values in [-300, 300] plus sampling of full range near boundary points.
vals = []
# scan half the FP16 grid
for e in range(-14, 16):
    for m in range(1, 1024):
        v = m * 2.0**e
        vals.append(v)
        vals.append(-v)
vals = sorted(set(vals))
vals = [v for v in vals if abs(v) <= 65504 and v != 0]
xs = torch.tensor(vals, dtype=torch.float32)
t = xs * (2.0/math.pi) + 0.5
dist = (t - torch.round(t)).abs()
# pick the FP16 values closest to a reduction boundary
idx = torch.argsort(dist)[:2048]
cand = vals[:0]
sel_vals = xs[idx].tolist()
# also include boundary neighborhood at large magnitude for exponent coverage
big = [v for v in vals if 1024 <= abs(v) <= 65504]
sel_vals += big[:4096]

x = torch.tensor(sel_vals, dtype=torch.float16, device=dev)
out = kernel.run(x)
ref = torch.sin(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
viol = err > 1e-6
res = {
    "tolerance": 1e-6,
    "n": int(x.numel()),
    "min_boundary_dist": float(dist.min().item()),
    "n_violations": int(viol.sum().item()),
    "max_err": float(err.max().item()),
    "n_big_violations": int(viol[len(sel_vals)-4096:].sum().item()) if len(sel_vals) > 4096 else 0,
    "n_nearboundary_violations": int(viol[:len(sel_vals)-4096].sum().item()) if len(sel_vals) > 4096 else int(viol.sum().item()),
}
if viol.any():
    vi = torch.nonzero(viol).flatten()
    res["example_violations"] = [{"x": float(x[i].item()), "err": float(err[i].item()), "actual": float(out[i].item()), "ref": float(ref[i].item())} for i in vi[:10].tolist()]
print(json.dumps(res))
