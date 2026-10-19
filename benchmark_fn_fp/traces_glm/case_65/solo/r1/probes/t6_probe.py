import torch, sys, json
sys.path.insert(0, '/root/cases/case_138')
import kernel

# All 65536 FP16 bit patterns; keep finite ones (exclude NaN/Inf which are out of domain)
bits = torch.arange(65536, dtype=torch.int32)
fp16 = bits.to(torch.uint16).view(torch.float16).to(torch.int16)  # reinterpret
all_vals = bits.to(torch.uint16).view(torch.float16)  # view bit patterns as fp16
finite_mask = torch.isfinite(all_vals)
x_all = all_vals[finite_mask].to('cuda')  # 65504 finite values
print("finite count:", x_all.numel())

out = kernel.run(x_all)  # fp32 on cuda
# reference: real-arithmetic rho = sqrt(1+x^2)-1 computed in FP64 on exact stored values
xd = x_all.to(torch.float64)
sq = xd * xd
rho = torch.sqrt(1.0 + sq) - 1.0

err = (out.to(torch.float64) - rho).abs()
tol = 1e-8 + 1e-5 * rho.abs()
ok = err <= tol
n_bad = int((~ok).sum())
print("violations:", n_bad)
print("max err:", float(err.max()), "max tol:", float(tol.max()))
print("worst margin (err-tol) max:", float((err-tol).max()))
if n_bad:
    idx = (~ok).nonzero().flatten()[:10]
    print("bad samples:", x_all[idx].tolist(), err[idx].tolist(), tol[idx].tolist())
print(json.dumps({"metric":"elementwise tolerance check, exhaustive finite FP16 domain",
 "finite_count": x_all.numel(), "violations": n_bad,
 "max_abs_err": float(err.max()), "worst_margin": float((err-tol).max())}))
