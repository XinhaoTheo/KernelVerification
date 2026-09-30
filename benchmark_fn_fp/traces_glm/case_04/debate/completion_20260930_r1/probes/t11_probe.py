import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_04/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
eps = 1e-6
n_cols = 4096
rows = [torch.randn(n_cols, device=dev, dtype=torch.float32) * s for s in [1e-3,1e-4,1e-6,1e-8,1e-12,1e-16,1e-18]]
X = torch.cat(rows).reshape(len(rows), n_cols).contiguous()
Y = kern.rms_norm_forward(X, eps)
X32 = X.float()
ms = (X32*X32).sum(dim=1)/n_cols
rstd_ref = 1.0/torch.sqrt(ms+eps)
Yref = X32*rstd_ref[:,None]
abs_err = (Y-Yref).abs().max().item()
rel_err = ((Y-Yref).abs()/Yref.abs().clamp_min(1e-38)).max().item()
rstd_k = (Y/X32).mean(dim=1)
# ulp in fp32 via math.ulp on python floats
ulps = [abs(rstd_k[i].item()-rstd_ref[i].item())/math.ulp(rstd_ref[i].item()) for i in range(len(rows))]
ac5 = bool(torch.allclose(Y, Yref, rtol=1e-5, atol=0))
ac6 = bool(torch.allclose(Y, Yref, rtol=1e-6, atol=0))
ac7 = bool(torch.allclose(Y, Yref, rtol=1e-7, atol=0))
print(json.dumps({"max_abs_err": abs_err, "max_rel_err": rel_err,
 "per_row_rstd_ulp_diff": ulps, "max_rstd_ulp_diff": max(ulps),
 "allclose_rtol_1e-5": ac5, "allclose_rtol_1e-6": ac6, "allclose_rtol_1e-7": ac7,
 "eps": eps, "n_cols": n_cols, "device": dev}))