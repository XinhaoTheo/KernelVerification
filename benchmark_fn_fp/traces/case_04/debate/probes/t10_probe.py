
import json, sys, torch
sys.path.insert(0, '/root/cases/case_04')
from kernel import rms_norm_forward

torch.manual_seed(0)
eps = 1e-6
out = {}
for name, dt in [('bf16', torch.bfloat16), ('fp16', torch.float16)]:
    X = torch.randn(64, 512, device='cuda').to(dt)
    Y = rms_norm_forward(X, eps).float()          # kernel output (fp32)
    x32 = X.float()
    rstd = torch.rsqrt((x32*x32).mean(-1, keepdim=True) + eps)
    ref_fp32 = x32 * rstd                          # fp32 accumulation, fp32 store
    ref_lowp = ref_fp32.to(dt).float()             # dtype-matched reference (Liger stores in X.dtype)

    def stats(a, b):
        d = (a-b).abs()
        rel = d / b.abs().clamp_min(1e-30)
        return {'max_abs': d.max().item(), 'max_rel': rel.max().item(),
                'mean_rel': rel.mean().item(),
                'n_exact_eq': int((a == b).sum().item()), 'n_elem': a.numel()}

    out[name] = {
        'vs_dtype_matched_lowp_ref': stats(Y, ref_lowp),
        'vs_fp32_ref': stats(Y, ref_fp32),
        'allclose_rtol1e-3_vs_lowp_ref': bool(torch.allclose(Y, ref_lowp, rtol=1e-3, atol=0)),
        'allclose_rtol1e-2_vs_lowp_ref': bool(torch.allclose(Y, ref_lowp, rtol=1e-2, atol=1e-3)),
        'allclose_rtol1e-5_vs_fp32_ref': bool(torch.allclose(Y, ref_fp32, rtol=1e-5, atol=1e-6)),
    }
# also fp32 input sanity
Xf = torch.randn(64, 512, device='cuda')
Yf = rms_norm_forward(Xf, eps)
reff = Xf * torch.rsqrt((Xf*Xf).mean(-1, keepdim=True) + eps)
out['fp32_input'] = {'max_abs': (Yf-reff).abs().max().item(),
                     'max_rel': ((Yf-reff).abs()/reff.abs().clamp_min(1e-30)).max().item()}
print(json.dumps(out))
