
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(X, eps):
    x = X.float()
    ms = (x*x).sum(-1, keepdim=True)/x.shape[-1]
    return x*torch.rsqrt(ms+eps)

torch.manual_seed(0)
res = []
cases = []
cases.append(("fp32_std", torch.randn(8, 4096, device='cuda')))
cases.append(("fp32_nonpow2", torch.randn(5, 1000, device='cuda')))
cases.append(("bf16_rt", torch.randn(8, 4096, device='cuda').bfloat16().float()))
cases.append(("bf16_dtype", torch.randn(8, 1023, device='cuda').bfloat16()))
tiny = torch.randn(4, 512, device='cuda')*1e-20
cases.append(("near_zero", tiny))
z = torch.zeros(3, 300, device='cuda')
cases.append(("zeros", z))
mix = torch.randn(4, 777, device='cuda')
mix[0] *= 1e-18; mix[1] *= 1e8
cases.append(("mixed_mag", mix))
for name, X in cases:
    eps = 1e-6
    Y = m.rms_norm_forward(X, eps)
    R = ref(X, eps)
    d = (Y-R).abs()
    denom = R.abs().clamp_min(1e-30)
    res.append(dict(case=name, shape=list(X.shape), dtype=str(X.dtype),
                    out_dtype=str(Y.dtype),
                    max_abs=float(d.max()), max_rel=float((d/denom).max()),
                    finite=bool(torch.isfinite(Y).all()),
                    ref_max=float(R.abs().max())))
print(json.dumps(dict(metric="max abs/rel err vs fp32 reference", results=res), indent=1))
