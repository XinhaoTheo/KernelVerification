
import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(X, eps):
    x = X.float()
    ms = (x*x).sum(-1, keepdim=True)/x.shape[-1]
    return x*torch.rsqrt(ms+eps)

torch.manual_seed(3)
rows, cols = 4, 512
base = torch.randn(rows, cols, device='cuda')

out = []
# scan scales and eps values, including sub-normal-triggering scales
for scale in [1e-1, 1e-5, 1e-10, 1e-18, 1e-20, 1e-22, 1e-25, 1e-30, 1e-38, 0.0]:
    for eps in [1e-6, 1e-5, 1e-8, 1e-12, 1e-20, 1e-30, 1e-38, 1e-42, 0.0]:
        X = base*scale
        Y = m.rms_norm_forward(X, eps)
        R = ref(X, eps)
        fy = bool(torch.isfinite(Y).all()); fr = bool(torch.isfinite(R).all())
        d = (Y-R).abs()
        den = R.abs().clamp_min(1e-30)
        mr = float((d/den).max()) if fy and fr else float('nan')
        if (not fy) or fy != fr or (fy and fr and mr > 1e-4):
            out.append(dict(scale=scale, eps=eps, fin_y=fy, fin_r=fr, max_rel=mr))

# boundary: smallest eps>0 that keeps kernel finite at scale 1e-22
bnd = []
X = base*1e-22
for eps in [0.0, 1e-45, 1e-42, 1e-40, 1e-38, 1e-36, 1e-30, 1e-20, 1e-12, 1e-6]:
    Y = m.rms_norm_forward(X, eps)
    bnd.append(dict(eps=eps, fin=bool(torch.isfinite(Y).all())))

# also: bf16 round-trip tiny rows with realistic eps
Xb = (base*1e-20).bfloat16()
Yb = m.rms_norm_forward(Xb, 1e-6); Rb = ref(Xb, 1e-6)
bf = dict(fin=bool(torch.isfinite(Yb).all()),
          max_abs=float((Yb-Rb).abs().max()))

print(json.dumps(dict(metric="finiteness + max rel err vs fp32 ref over eps x scale grid",
                      failures=out, n_fail=len(out), boundary_scale_1e_22=bnd, bf16_tiny=bf)))
