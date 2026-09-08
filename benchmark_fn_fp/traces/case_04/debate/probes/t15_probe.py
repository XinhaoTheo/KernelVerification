
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
eps = 1e-6
out = {}
def stats(a,b):
    d=(a-b).abs(); rel=d/b.abs().clamp_min(1e-20)
    return {"max_abs": d.max().item(), "max_rel": rel.max().item(),
            "n_nonfinite_kernel": int((~torch.isfinite(a)).sum().item()),
            "n_nonfinite_ref": int((~torch.isfinite(b)).sum().item())}
cases = {
  "all_zero_rows": torch.zeros(4,256,device="cuda"),
  "tiny_1e-8": torch.full((4,256),1e-8,device="cuda"),
  "tiny_1e-20": torch.full((4,256),1e-20,device="cuda"),
  "denormal_1e-40": torch.full((4,256),1e-40,device="cuda"),
  "mixed_small": (torch.randn(4,256,device="cuda")*1e-6),
  "large_1e20": torch.full((4,256),1e20,device="cuda"),
}
for name,X in cases.items():
    Y = m.rms_norm_forward(X, eps)
    ms = (X*X).sum(-1,keepdim=True)/X.shape[-1]
    ref = X*torch.rsqrt(ms+eps)
    s = stats(Y, ref)
    s["kernel_sample"] = Y[0,0].item(); s["ref_sample"] = ref[0,0].item()
    out[name]=s
print(json.dumps(out))
