import sys, json, torch
sys.path.insert(0, "/root/cases/case_138")
import kernel as K

xs = torch.tensor([65504.0, -65504.0, 65503.0, 65503.5, 60000.0, 5.0e4, 3.0e4, 1.0e4], dtype=torch.float16, device="cuda")
out = K.run(xs)
ref = K.reference(xs.cpu().cuda() if xs.is_cuda else xs)
# recompute reference in fp64 on cuda
r64 = xs.to(torch.float64); sq = r64*r64; rho = torch.sqrt(1.0+sq)-1.0
err = (out.to(torch.float64)-rho).abs()
tol = 1e-8 + 1e-5*rho.abs()
print(json.dumps({"x": xs.to(torch.float64).tolist(), "out": out.tolist(), "rho": rho.tolist(), "err": err.tolist(), "tol": tol.tolist(), "violations": int((err>tol).sum())}))
