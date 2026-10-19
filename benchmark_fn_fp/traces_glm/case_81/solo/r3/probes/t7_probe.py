import torch, json, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(123)

# Generate random fp16 rows, look for severe cancellation in the real or imag numerator
M = 4_000_000
A = (torch.empty((M,), dtype=torch.float32).uniform_(-65504, 65504, generator=g)).to(torch.float16)
B = (torch.empty((M,), dtype=torch.float32).uniform_(-65504, 65504, generator=g)).to(torch.float16)
C = (torch.empty((M,), dtype=torch.float32).uniform_(-65504, 65504, generator=g)).to(torch.float16)
D = (torch.empty((M,), dtype=torch.float32).uniform_(-65504, 65504, generator=g)).to(torch.float16)

a,b,c,d = [t.double() for t in (A,B,C,D)]
rn = a*c + b*d
inum = b*c - a*d
denom = rn.abs() / (a.abs()*c.abs() + b.abs()*d.abs())
denim = inum.abs() / (b.abs()*c.abs() + a.abs()*d.abs())
score = torch.minimum(denom, denim)
score = torch.where((rn!=0) & (inum!=0), score, torch.full_like(score, 1e9))
idx = torch.argsort(score)[:100000]

num = torch.stack([A[idx], B[idx]], dim=1).to(dev)
den = torch.stack([C[idx], D[idx]], dim=1).to(dev)
out = kernel.run(num, den)
ref = kernel.reference(num.cpu(), den.cpu()).to(dev).double()
err = (out.double() - ref).abs()
tol = 1e-7 + 5e-7*ref.abs()
ratio = err/tol
viol = ratio > 1.0
res = {
 "n": num.shape[0],
 "min_cancellation_ratio": float(score[idx].min()),
 "max_ratio": float(ratio.max()),
 "violations": int(viol.sum()),
 "output_finite": bool(torch.isfinite(out).all()),
}
if int(viol.sum())>0:
    i = int(ratio.flatten().argmax())
    r,cc = divmod(i,2)
    res["worst_row"] = [float(num[r,0]), float(num[r,1]), float(den[r,0]), float(den[r,1]), "real" if cc==0 else "imag"]
    res["worst_actual"] = float(out[r,cc]); res["worst_target"] = float(ref[r,cc])
# benign check
n3,d3 = kernel.make_inputs(dev, size=4093, seed=0)
out3 = kernel.run(n3,d3)
ref3 = kernel.reference(n3.cpu(), d3.cpu()).to(dev).double()
r3 = ((out3.double()-ref3).abs()/(1e-7+5e-7*ref3.abs()))
res["benign_max_ratio"] = float(r3.max())
print(json.dumps(res))
