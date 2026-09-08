
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(X, eps):
    x = X.double()
    ms = (x*x).sum(-1, keepdim=True)/x.shape[-1]
    return (x * torch.rsqrt(ms+eps)).float()

res = {}
torch.manual_seed(1)
eps = 1e-6
for n in [1,2,3,17,64,4096,8192,16384,32768]:
    X = torch.randn(4,n, device='cuda')
    try:
        Y = m.rms_norm_forward(X, eps)
        R = ref(X, eps)
        err = (Y.float()-R).abs()
        res[f"fp32_n{n}"] = dict(max_abs=err.max().item(),
            max_rel=(err/R.abs().clamp_min(1e-30)).max().item(),
            out_dtype=str(Y.dtype))
    except Exception as e:
        res[f"fp32_n{n}"] = dict(error=repr(e)[:200])

# bf16 near-zero rows
for scale in [1e-3, 1e-6, 1e-20]:
    X = (torch.randn(8,256, device='cuda')*scale).bfloat16()
    try:
        Y = m.rms_norm_forward(X, eps)
        R = ref(X, eps)
        err = (Y.float()-R).abs()
        res[f"bf16_scale_{scale}"] = dict(max_abs=err.max().item(),
            ref_absmax=R.abs().max().item(),
            max_rel=(err/R.abs().clamp_min(1e-30)).max().item(),
            nan=int(torch.isnan(Y).sum()))
    except Exception as e:
        res[f"bf16_scale_{scale}"] = dict(error=repr(e)[:200])

# non-contiguous-ish: slice of wider tensor (row stride != n_cols) -- note kernel calls X.float() which makes contiguous
big = torch.randn(8, 512, device='cuda')
Xv = big[:, :300]
try:
    Y = m.rms_norm_forward(Xv, eps)
    R = ref(Xv, eps)
    res["sliced_view_300"] = dict(max_abs=(Y.float()-R).abs().max().item())
except Exception as e:
    res["sliced_view_300"] = dict(error=repr(e)[:200])

print(json.dumps(res))
