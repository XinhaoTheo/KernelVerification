import torch, json, sys
sys.path.insert(0, "/root/cases/case_12")
from kernel import rms_norm_forward

def ref(X, eps):
    ms = (X * X).sum(dim=1, dtype=torch.float32) / X.shape[1]
    rstd = 1.0 / torch.sqrt(ms + eps)
    return X * rstd.unsqueeze(1), rstd

eps = 1e-6
# rows: all-zero, tiny (1e-8 scale), normal randn
X = torch.cat([
    torch.zeros(1, 64),
    torch.full((1, 64), 1e-8),
    torch.randn(4, 64),
]).cuda()

Y = rms_norm_forward(X, eps)
Yr, rstd_ref = ref(X, eps)
# kernel rstd implied from Y/X where nonzero
X32 = X.float()
ratio = (Y.float() / X32).nan_to_num(0)  # zero rows give 0/0
# compute kernel rstd analytically for the all-zero row: 1/(0+eps)=1e6 vs ref 1/sqrt(eps)=1e3
k_rstd_zero = 1.0 / (0.0 + eps)
ref_rstd_zero = 1.0 / (eps ** 0.5)
d = {
    "kernel_impl": "1/(sqrt(ms)+eps)",
    "eps": eps,
    "max_abs_err_normal_rows": float((Y[2:] - Yr[2:]).abs().max()),
    "kernel_rstd_zero_row": k_rstd_zero,
    "ref_rstd_zero_row": ref_rstd_zero,
    "rstd_ratio_zero_row": k_rstd_zero / ref_rstd_zero,
    "Y_zero_row_max_abs": float(Y[0].abs().max()),
    "Y_tiny_row_max_abs_kernel": float(Y[1].abs().max()),
    "Y_tiny_row_max_abs_ref": float(Yr[1].abs().max()),
    "tiny_row_out_ratio": float(Y[1].abs().max() / Yr[1].abs().max()),
    "max_rel_err_all_rows": float(((Y.float() - Yr).abs() / Yr.abs().clamp_min(1e-30)).max()),
}
print(json.dumps(d))