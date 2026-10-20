import sys, json, numpy as np, torch
sys.path.insert(0, "/root/cases/case_97")
import kernel

# All distinct finite FP16 values in [-10,10]
vals = torch.tensor(np.arange(65536, dtype=np.uint16).view(np.float16), dtype=torch.float16)
vals = vals[torch.isfinite(vals) & (vals.abs() <= 10.0)]
x = vals.to("cuda")
out = kernel.run(x)
ref = kernel.reference(x.cpu()).to("cuda")
err = (out.double() - ref.double()).abs()
tol = 1e-7 + 1e-6 * ref.double().abs()

# c1 focus: k=0 region, |x| just under ln2/2 (~0.3466)
ln2_2 = 0.6931471805599453 / 2
mask0 = ((x.abs() <= ln2_2 + 1e-4) & (x.abs() >= 0.3))
viol = err > tol
worst_idx = int(torch.argmax(torch.where(mask0, err / tol, torch.zeros_like(err))))
xw = float(x[worst_idx])
print(json.dumps({
    "metric": "abs err vs 1e-7+1e-6*|target| over all 9000+ legal FP16 values, focus k=0 max|r| region",
    "n_legal_fp16_values": int(x.numel()),
    "k0_region_violations": int((viol & mask0).sum()),
    "total_violations": int(viol.sum()),
    "worst_k0_region": {"x": xw, "err": float(err[worst_idx]), "tol": float(tol[worst_idx]), "ratio": float(err[worst_idx]/tol[worst_idx])},
    "global_worst_ratio": float((err/tol).max()),
    "global_worst_x": float(x[int(torch.argmax(err/tol))]),
}))