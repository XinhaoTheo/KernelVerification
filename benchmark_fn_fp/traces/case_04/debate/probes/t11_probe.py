
import json, sys, torch
sys.path.insert(0, '/root/cases/case_04')
from kernel import rms_norm_forward

eps = 1e-6
torch.manual_seed(0)
rows = {
    'normal':      torch.randn(1, 512),
    'tiny_1e-22':  torch.randn(1, 512) * 1e-22,
    'tiny_1e-30':  torch.randn(1, 512) * 1e-30,
    'denormal':    torch.full((1, 512), 1e-40),
    'all_zero':    torch.zeros(1, 512),
    'small_1e-4':  torch.randn(1, 512) * 1e-4,
}
names = list(rows)
X = torch.cat([rows[n] for n in names], 0).cuda()
Y = rms_norm_forward(X, eps).float()

x32 = X.float()
ref32 = x32 * torch.rsqrt((x32*x32).mean(-1, keepdim=True) + eps)
x64 = X.double()
ref64 = (x64 * torch.rsqrt((x64*x64).mean(-1, keepdim=True) + eps)).float()

res = {}
for i, n in enumerate(names):
    y, r32, r64 = Y[i], ref32[i], ref64[i]
    res[n] = {
        'row_max_abs_x': x32[i].abs().max().item(),
        'kernel_max_abs_y': y.abs().max().item(),
        'ref_fp32_max_abs_y': r32.abs().max().item(),
        'ref_fp64_max_abs_y': r64.abs().max().item(),
        'max_abs_err_vs_fp32ref': (y-r32).abs().max().item(),
        'max_abs_err_vs_fp64ref': (y-r64).abs().max().item(),
        'max_rel_err_vs_fp64ref': ((y-r64).abs()/r64.abs().clamp_min(1e-38)).max().item(),
        'exact_match_fp32ref': bool(torch.equal(y, r32)),
        'n_nonfinite_kernel': int((~torch.isfinite(y)).sum().item()),
    }
res['_meta'] = {'eps': eps, 'shape': list(X.shape), 'dtype': str(X.dtype)}
print(json.dumps(res))
