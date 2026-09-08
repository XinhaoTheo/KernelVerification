
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(X, eps):
    x = X.float()
    ms = (x*x).sum(-1, keepdim=True)/x.shape[-1]
    return x*torch.rsqrt(ms+eps)

torch.manual_seed(1)
out=[]
worst=0.0
for rows, cols in [(1,1),(2,3),(3,17),(7,127),(4,4096),(2,8192),(64,2048),(129,769)]:
    for eps in [1e-6, 1e-5, 0.0, 1e-2]:
        for scale, tag in [(1.0,'unit'),(1e-20,'tiny'),(1e8,'huge')]:
            X = (torch.randn(rows, cols, device='cuda')*scale)
            Y = m.rms_norm_forward(X, eps)
            R = ref(X, eps)
            d = (Y-R).abs()
            den = R.abs().clamp_min(1e-30)
            ma=float(d.max()); mr=float((d/den).max())
            fin_y=bool(torch.isfinite(Y).all()); fin_r=bool(torch.isfinite(R).all())
            if (ma>0 and mr>1e-5) or fin_y!=fin_r:
                out.append(dict(rows=rows,cols=cols,eps=eps,tag=tag,max_abs=ma,max_rel=mr,fin_y=fin_y,fin_r=fin_r))
            worst=max(worst,mr if ma>0 else 0.0)

# bf16 round-trip consistency, sweeping widths
bf=[]
for cols in [1,15,64,1000,4096]:
    Xb = torch.randn(6, cols, device='cuda').bfloat16()
    Y = m.rms_norm_forward(Xb, 1e-6); R = ref(Xb, 1e-6)
    bf.append(dict(cols=cols, max_abs=float((Y-R).abs().max()),
                   max_rel=float(((Y-R).abs()/R.abs().clamp_min(1e-30)).max()),
                   out_dtype=str(Y.dtype)))

# repeated-call determinism
Xd = torch.randn(8, 3000, device='cuda')
Y1 = m.rms_norm_forward(Xd, 1e-6); Y2 = m.rms_norm_forward(Xd, 1e-6)
det = bool(torch.equal(Y1, Y2))

print(json.dumps(dict(metric="max rel err vs fp32 ref over shape/eps/scale sweep",
                      mismatches=out, worst_rel=worst, bf16=bf, deterministic=det)))
