
import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(1)
eps = 1e-6
n = 64
base = torch.randn(n, device='cuda', dtype=torch.float32)
scales = [1e-3, 1e-20, 1e-30, 1e-40]
rows = [base*s for s in scales] + [torch.zeros(n, device='cuda')]
X = torch.stack(rows).contiguous()
labels = [f"{s:g}" for s in scales] + ["zero"]

Y = m.rms_norm_forward(X, eps)
Xd = X.double()
msd = (Xd*Xd).sum(dim=1, keepdim=True)/n
rstd_ref = 1.0/torch.sqrt(msd + eps)
Yref = Xd * rstd_ref

res = {"eps": eps, "n_cols": n, "rows": {}}
for i, lab in enumerate(labels):
    yk = Y[i].double(); yr = Yref[i]
    nz = yr.abs() > 0
    rel = torch.zeros_like(yr)
    rel[nz] = (yk[nz]-yr[nz]).abs()/yr[nz].abs()
    # implied rstd via least squares (fp64) to average out fp32 rounding
    denom = (Xd[i]*Xd[i]).sum().item()
    implied = ((Y[i].double()*Xd[i]).sum()/denom).item() if denom > 0 else float('nan')
    res["rows"][lab] = {
        "input_min_abs": X[i].abs().min().item(),
        "input_max_abs": X[i].abs().max().item(),
        "input_subnormal_count": int(((X[i].abs() > 0) & (X[i].abs() < 1.1754944e-38)).sum().item()),
        "mean_square_fp64": msd[i].item(),
        "rstd_ref_fp64": rstd_ref[i].item(),
        "rstd_implied_kernel": implied,
        "rstd_rel_err": abs(implied-rstd_ref[i].item())/rstd_ref[i].item() if denom>0 else None,
        "max_rel_err_y": rel.max().item(),
        "exact_zeros_where_ref_nonzero": int(((Y[i]==0) & nz).sum().item()),
        "n_ref_nonzero": int(nz.sum().item()),
        "nan_count": int(torch.isnan(Y[i]).sum().item()),
        "inf_count": int(torch.isinf(Y[i]).sum().item()),
        "y_max_abs": Y[i].abs().max().item(),
        "yref_max_abs": yr.abs().max().item(),
    }
res["global_nan"] = int(torch.isnan(Y).sum().item())
res["global_inf"] = int(torch.isinf(Y).sum().item())
print(json.dumps(res))
