import torch, json, sys
sys.path.insert(0, "/root/cases/case_137")
from kernel import run, reference

vals = [0.0, -0.0]
for i in range(1, 3001):
    vals += [i * (2.0 ** -24), -i * (2.0 ** -24)]
x = torch.tensor(vals, dtype=torch.float16, device="cuda")
out = run(x).cpu().double()
ref64 = reference(x.cpu())
err = (out - ref64).abs()
tol = 1e-6
res = {
    "n": x.numel(),
    "max_abs_err": float(err.max()),
    "tolerance": tol,
    "n_violations": int((err > tol).sum()),
    "err_at_pos_zero": float(err[0]),
    "err_at_neg_zero": float(err[1]),
    "output_at_pos_zero": float(out[0]),
    "target_ln2": 0.6931471805599453,
}
print(json.dumps(res))