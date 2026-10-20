import sys, json, numpy as np, torch
sys.path.insert(0, "/root/cases/case_97")
import kernel

vals = torch.tensor(np.arange(65536, dtype=np.uint16).view(np.float16), dtype=torch.float16)
vals = vals[torch.isfinite(vals) & (vals.abs() <= 10.0)]
x = vals.to("cuda")
out = kernel.run(x)
ref = kernel.reference(x.cpu()).to("cuda")
err = (out.double() - ref.double()).abs()
tol = 1e-7 + 1e-6 * ref.double().abs()

# c2 focus: large positive inputs, k=14, x >= 9.0
mask = x >= 9.0
viol = err > tol
sub = err[mask] / tol[mask]
i = 0
xlist = x[mask].cpu()
worst = int(torch.argmax(sub)) if sub.numel() else 0
print(json.dumps({
    "metric": "abs err vs 1e-7+1e-6*|target|, focus x>=9 (k=14) region",
    "n_x_geq_9": int(mask.sum()),
    "violations_x_geq_9": int((viol & mask).sum()),
    "total_violations_all_legal": int(viol.sum()),
    "worst_x_geq_9": {"x": float(xlist[worst]), "err": float(err[mask][worst]), "tol": float(tol[mask][worst]), "ratio": float(sub[worst])},
    "x_plus_10": {"err": float(err[x == 10.0]), "tol": float(tol[x == 10.0]), "ratio": float(err[x == 10.0]/tol[x == 10.0])},
}))