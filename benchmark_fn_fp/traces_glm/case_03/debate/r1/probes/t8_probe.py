import torch, json, importlib.util, os, math
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_04/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
eps = 1e-6
n_rows, n_cols = 64, 4096
# near-zero rows: magnitudes such that mean_square ~ eps and smaller
rows = []
for scale in [1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-16, 1e-18]:
    rows.append(torch.randn(n_cols, device=dev, dtype=torch.float32) * scale)
X = torch.cat(rows).reshape(len(rows), n_cols).contiguous()
Y = kern.rms_norm_forward(X, eps)
# fp32 reference: exact formula as in problem.txt, fp32 accumulation
X32 = X.float()
ms = (X32 * X32).sum(dim=1) / n_cols
rstd_ref = 1.0 / torch.sqrt(ms + eps)
Yref = X32 * rstd_ref[:, None]
# also direct rstd from kernel? recompute ms as kernel does for ulp comparison
abs_err = (Y - Yref).abs().max().item()
rel_err = ((Y - Yref).abs() / Yref.abs().clamp_min(1e-38)).max().item()
# ulp-level check of rstd: compare torch's 1/sqrt vs rsqrt intrinsic via kernel output scaling
# infer kernel rstd: y/x on non-zero entries
rstd_k = (Y / X32).mean(dim=1)  # rstd constant per row
ulp = torch.abs(rstd_k - rstd_ref) / torch.spacing(rstd_ref.abs())
allclose_1e5 = torch.allclose(Y, Yref, rtol=1e-5, atol=0)
allclose_1e6 = torch.allclose(Y, Yref, rtol=1e-6, atol=0)
print(json.dumps({"max_abs_err": abs_err, "max_rel_err": rel_err,
 "max_rstd_ulp_diff": ulp.max().item(), "mean_rstd_ulp_diff": ulp.mean().item(),
 "allclose_rtol_1e-5": bool(allclose_1e5), "allclose_rtol_1e-6": bool(allclose_1e6),
 "row_scales": [1e-3,1e-4,1e-6,1e-8,1e-12,1e-16,1e-18], "eps": eps,
 "mean_square_range": [ms.min().item(), ms.max().item()], "device": dev}))