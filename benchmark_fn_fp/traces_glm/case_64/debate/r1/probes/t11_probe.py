import torch, json, sys
sys.path.insert(0, "/root/cases/case_137")
from kernel import run, reference

vals = []
neg = -0.0005
while neg >= -25.0:
    vals.append(neg)
    neg -= 0.0005
vals += [-20.0, -25.0, -30.0, -40.0, -100.0, -1000.0, -10000.0, -65504.0,
         -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0, -18.0, -19.0]
x = torch.tensor(vals, dtype=torch.float16, device="cuda")
out = run(x).cpu().double()
ref64 = reference(x.cpu())
err = (out - ref64).abs()
tol = 1e-6
i = int(err.argmax())
res = {
    "n": x.numel(),
    "max_abs_err": float(err.max()),
    "tolerance": tol,
    "n_violations": int((err > tol).sum()),
    "worst_x": float(x[i].cpu()),
    "out_at_worst": float(out[i]),
    "ref_at_worst": float(ref64[i]),
}
print(json.dumps(res))