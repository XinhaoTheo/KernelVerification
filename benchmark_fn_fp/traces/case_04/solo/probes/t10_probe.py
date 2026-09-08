
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref64(X, eps):
    X64 = X.double()
    ms = (X64*X64).sum(dim=1, keepdim=True)/X64.shape[1]
    return X64 * torch.rsqrt(ms + eps)

torch.manual_seed(1)
out = {}
eps = 1e-6
shapes = [(1,1),(1,3),(7,5),(32,31),(32,33),(128,2048),(16,4096),(8,8192),(4,16384),(3,1000),(2048,64)]
worst = 0.0
for (r,c) in shapes:
    X = torch.randn(r,c, device="cuda")
    try:
        Y = k.rms_norm_forward(X, eps)
        R = ref64(X, eps)
        rel = ((Y.double()-R).abs()/R.abs().clamp_min(1e-300)).max().item()
        out[f"{r}x{c}"] = dict(max_rel_vs_fp64=rel, out_dtype=str(Y.dtype))
        worst = max(worst, rel)
    except Exception as e:
        out[f"{r}x{c}"] = dict(error=repr(e))

# dtype variants
for dt in [torch.float32, torch.bfloat16, torch.float16]:
    X = (torch.randn(64,512, device="cuda")).to(dt)
    Y = k.rms_norm_forward(X, eps)
    R = ref64(X.float(), eps)
    out[f"dtype_{dt}"] = dict(max_rel_vs_fp64=((Y.double()-R).abs()/R.abs().clamp_min(1e-300)).max().item(),
                              out_dtype=str(Y.dtype))

# determinism
X = torch.randn(256,1024, device="cuda")
Y1 = k.rms_norm_forward(X, eps); Y2 = k.rms_norm_forward(X, eps)
out["determinism_max_abs_diff"] = float((Y1-Y2).abs().max())

# scaling invariance sanity (eps=0): y should be scale-invariant
X = torch.randn(32,256, device="cuda")
Ya = k.rms_norm_forward(X, 0.0); Yb = k.rms_norm_forward(X*1024.0, 0.0)
out["scale_invariance_max_abs"] = float((Ya-Yb).abs().max())
out["worst_rel_vs_fp64_shapes"] = worst
print(json.dumps(out, indent=1))
