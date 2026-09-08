
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
eps = 1e-6
X = torch.randn(8, 1024, device="cuda").to(torch.bfloat16)
Y = m.rms_norm_forward(X, eps).float()
# reference: fp32 accumulation, then bf16 store (Liger behavior)
x32 = X.float()
ms = (x32*x32).sum(-1, keepdim=True)/x32.shape[-1]
rstd = torch.rsqrt(ms + eps)
ref_unrounded = x32*rstd
ref_bf16 = ref_unrounded.to(torch.bfloat16).float()
def stats(a, b):
    d = (a-b).abs()
    rel = d/ b.abs().clamp_min(1e-12)
    return {"max_abs": d.max().item(), "max_rel": rel.max().item(),
            "n_diff": int((a!=b).sum().item()), "n_total": a.numel()}
out = {"vs_bf16_stored_reference": stats(Y, ref_bf16),
       "vs_fp32_unrounded_reference": stats(Y, ref_unrounded)}
# does kernel output, if downcast to bf16, match the bf16 reference exactly?
out["kernel_rounded_to_bf16_matches_ref"] = bool(torch.equal(Y.to(torch.bfloat16), ref_bf16.to(torch.bfloat16)))
# fp32 input path sanity
Xf = torch.randn(8, 1024, device="cuda")
Yf = m.rms_norm_forward(Xf, eps)
msf = (Xf*Xf).sum(-1, keepdim=True)/Xf.shape[-1]
reff = Xf*torch.rsqrt(msf+eps)
out["fp32_input"] = stats(Yf, reff)
print(json.dumps(out))
