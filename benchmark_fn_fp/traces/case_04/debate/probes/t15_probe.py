
import json, sys, torch
sys.path.insert(0, '/root/cases/case_04')
from kernel import rms_norm_forward

torch.manual_seed(0)
eps = 1e-6
out = {}
for name, dt in [('bf16', torch.bfloat16), ('fp16', torch.float16)]:
    X = torch.randn(64, 512, device='cuda').to(dt)
    Y32 = rms_norm_forward(X, eps)              # fp32 kernel output
    Ylow = Y32.to(dt)                            # rounded to input dtype
    x32 = X.float()
    rstd = torch.rsqrt((x32*x32).mean(-1, keepdim=True) + eps)
    ref_low = (x32 * rstd).to(dt)                # Liger-style: fp32 accum, store in X.dtype
    eq = (Ylow == ref_low)
    d = (Ylow.float() - ref_low.float()).abs()
    out[name] = {
        'bitwise_equal_after_rounding': bool(torch.equal(Ylow, ref_low)),
        'n_mismatch_after_rounding': int((~eq).sum().item()),
        'n_elem': Ylow.numel(),
        'max_abs_err_after_rounding': d.max().item(),
        'max_ulp_like_rel_after_rounding': (d / ref_low.float().abs().clamp_min(1e-30)).max().item(),
    }

# fp32 vs bf16-round-trip consistency of the same logical tensor (problem.txt clause)
Xf = torch.randn(64, 512, device='cuda')
Xb = Xf.bfloat16()
Yf = rms_norm_forward(Xf, eps)
Yb = rms_norm_forward(Xb, eps)
refb = Xb.float() * torch.rsqrt((Xb.float()**2).mean(-1, keepdim=True) + eps)
out['roundtrip'] = {
    'kernel_bf16in_vs_fp32accum_ref_max_rel': ((Yb-refb).abs()/refb.abs().clamp_min(1e-30)).max().item(),
    'kernel_bf16in_vs_kernel_fp32in_max_rel': ((Yb-Yf).abs()/Yf.abs().clamp_min(1e-30)).max().item(),
    'note': 'second number reflects input quantization, not kernel error',
}
print(json.dumps(out))
