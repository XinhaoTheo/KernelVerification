import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
rows = [
 [65504, 0, 65504, 0],           # real=1 exact
 [65504, 0, 0, 65504],           # real=0, imag=-1 exact
 [65504, 65504, 65504, 65504],   # scale=2*65504^2 inexact
 [65504, 0, 65504, 6e-8],        # tiny subnormal d
 [65504, 65504, 65504, 6e-8],
 [60000, 30000, 60000, 30000],
 [65504, 65504, 1, 6e-8],        # tiny denominator
 [6e-8, 6e-8, 65504, 65504],
 [0.5, 0.25, 0.5, 0.25],
 [65504, -0.5, 1, 65504],
]
num = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.float16, device="cuda")
den = torch.tensor([[r[2], r[3]] for r in rows], dtype=torch.float16, device="cuda")
out = k.run(num, den).double()
ref = k.reference(num, den)
err = (out - ref).abs()
tol = 1e-7 + 5e-7 * ref.abs()
viol = err > tol
print("err:", err.tolist()); print("tol:", tol.tolist())
print(json.dumps({"rows_tested": len(rows), "n_violations": int(viol.sum()),
                  "max_violation_ratio": float((err / tol).max().item())}))