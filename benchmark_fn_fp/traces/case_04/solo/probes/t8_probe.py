
import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(X, eps):
    X32 = X.float()
    ms = (X32*X32).sum(dim=1, keepdim=True)/X32.shape[1]
    return X32 * torch.rsqrt(ms + eps)

torch.manual_seed(0)
res = {}
cases = {}
cases["fp32_randn_4096x1024"] = torch.randn(4096,1024, device="cuda")
cases["fp32_nonpow2_ncols_777"] = torch.randn(64,777, device="cuda")
x = torch.randn(64,1024, device="cuda")
cases["bf16_roundtrip"] = x.to(torch.bfloat16).float()
cases["bf16_native_input"] = x.to(torch.bfloat16)
cases["near_zero_1e-3"] = torch.randn(64,1024, device="cuda")*1e-3
cases["near_zero_1e-8"] = torch.randn(64,1024, device="cuda")*1e-8
cases["near_zero_1e-20"] = torch.randn(64,1024, device="cuda")*1e-20
cases["exact_zero_rows"] = torch.zeros(8,1024, device="cuda")
cases["large_1e18"] = torch.randn(16,1024, device="cuda")*1e18
cases["mixed_rows"] = torch.cat([torch.zeros(2,512,device="cuda"),
                                 torch.randn(2,512,device="cuda")*1e-25,
                                 torch.randn(2,512,device="cuda")*1e5], 0)

eps = 1e-6
for name, X in cases.items():
    try:
        Y = k.rms_norm_forward(X, eps)
        R = ref(X, eps)
        d = (Y.float()-R).abs()
        denom = R.abs().clamp_min(1e-30)
        res[name] = dict(shape=list(X.shape), in_dtype=str(X.dtype), out_dtype=str(Y.dtype),
                         max_abs=float(d.max()), max_rel=float((d/denom).max()),
                         ref_absmax=float(R.abs().max()),
                         y_nan=int(torch.isnan(Y).sum()), y_inf=int(torch.isinf(Y).sum()),
                         ref_nan=int(torch.isnan(R).sum()), ref_inf=int(torch.isinf(R).sum()))
    except Exception as e:
        res[name] = dict(error=repr(e))

# eps variation
for eps2 in [0.0, 1e-12, 1e-5, 1e-2]:
    X = torch.randn(32,512, device="cuda")
    Y = k.rms_norm_forward(X, eps2); R = ref(X, eps2)
    res[f"eps_{eps2}"] = dict(max_abs=float((Y-R).abs().max()), max_rel=float(((Y-R).abs()/R.abs().clamp_min(1e-30)).max()))

print(json.dumps(res, indent=1))
