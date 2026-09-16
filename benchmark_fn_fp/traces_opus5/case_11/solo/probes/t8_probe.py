
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_11/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref_inter(x, cos, sin):
    xe = x[..., 0::2].float(); xo = x[..., 1::2].float()
    c = cos.float().unsqueeze(1); s = sin.float().unsqueeze(1)
    o1 = xe*c - xo*s; o2 = xe*s + xo*c
    out = x.float().clone()
    out[..., 0::2] = o1; out[..., 1::2] = o2
    return out

res = []
torch.manual_seed(0)
for (S,H,D) in [(8,4,64),(3,2,8),(5,3,6),(4,2,48),(1,1,2),(16,8,128)]:
    for dt in [torch.float32, torch.float16]:
        try:
            x = torch.randn(S,H,D, device='cuda', dtype=dt)
            ang = torch.randn(S, D//2, device='cuda', dtype=torch.float32)
            cos = torch.cos(ang).to(dt); sin = torch.sin(ang).to(dt)
            out = m.apply_rotary(x, cos, sin, True)
            r = ref_inter(x, cos, sin)
            err = (out.float()-r).abs()
            res.append(dict(S=S,H=H,D=D,dtype=str(dt),max_abs=float(err.max()),
                            mean_abs=float(err.mean()), ok=bool(err.max() < (1e-4 if dt==torch.float32 else 5e-3))))
        except Exception as e:
            res.append(dict(S=S,H=H,D=D,dtype=str(dt),error=repr(e)[:300]))
print(json.dumps(dict(metric="max_abs_elementwise_err_vs_fp32_reference", cases=res), indent=1))
