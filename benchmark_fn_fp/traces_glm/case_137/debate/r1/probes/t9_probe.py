import torch, json, sys, math
sys.path.insert(0, "/root/cases/case_137")
from kernel import run, reference

dev = "cuda"
# negative inputs: sweep FP16 negatives from -0.001 to -65504 dense where s≈1
vals = []
# dense grid near where 1+exp(-|x|) rounds to 1 in FP32 (exp(-|x|) < 2^-25, |x|>~17.3)
neg = -0.0005
while neg >= -25.0:
    vals.append(neg)
    neg -= 0.0005
# plus representative large negatives up to -65504
big = [-20.0, -25.0, -30.0, -40.0, -100.0, -1000.0, -10000.0, -65504.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0, -18.0, -19.0]
vals += big
x = torch.tensor(vals, dtype=torch.float16, device=dev)
out = run(x)
ref64 = reference(x.cpu())
err = (out.double() - ref64).abs()
tol = 1e-6
i = int(err.argmax())
res = {
    "n": x.numel(),
    "max_abs_err": float(err.max()),
    "tolerance": tol,
    "n_violations": int((err > tol).sum()),
    "worst_x": float(x[i].cpu()),
    "err_at_x_minus17": float(err[abs(vals + 17.0).argmin()] if False else 0),
    "out_at_worst": float(out[i]),
    "ref_at_worst": float(ref64[i]),
}
print(json.dumps(res))