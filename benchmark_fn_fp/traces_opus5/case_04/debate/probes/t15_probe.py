
import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

out = {}
torch.manual_seed(3)
eps = 1e-6

def ref(x, eps):
    xf = x.float()
    ms = (xf*xf).sum(dim=1, keepdim=True)/xf.shape[1]
    return xf * torch.rsqrt(ms + eps)

# (a) bf16 round-trip stored back in fp32: values bf16-rounded, dtype fp32
X = torch.randn(8, 64, device='cuda', dtype=torch.float32)
Xrt = X.to(torch.bfloat16).float()
Yrt = m.rms_norm_forward(Xrt, eps)
R = ref(Xrt, eps)
out["roundtrip_fp32_dtype"] = str(Yrt.dtype)
out["roundtrip_max_rel_err"] = ((Yrt.double()-R.double()).abs()/R.double().abs().clamp_min(1e-30)).max().item()

# (b) non-contiguous fp32 input (transposed view) -- .float() is a no-op so stride leaks
Xt = torch.randn(64, 8, device='cuda', dtype=torch.float32).t()   # 8x64 view, stride (1,8)
out["nc_is_contiguous"] = Xt.is_contiguous()
out["nc_strides"] = list(Xt.stride())
try:
    Ync = m.rms_norm_forward(Xt, eps)
    Rnc = ref(Xt.contiguous(), eps)
    d = (Ync.double()-Rnc.double()).abs()
    out["nc_max_abs_err"] = d.max().item()
    out["nc_max_rel_err"] = (d/Rnc.double().abs().clamp_min(1e-30)).max().item()
    out["nc_frac_elems_wrong"] = (d > 1e-5*Rnc.double().abs().clamp_min(1e-30)).float().mean().item()
except Exception as e:
    out["nc_error"] = type(e).__name__ + ": " + str(e)[:200]

# (c) same logical data as non-contiguous bf16 view (copy path fixes stride?)
Xtb = Xt.to(torch.bfloat16)   # .to() on a view produces a contiguous tensor
out["nc_bf16_is_contiguous"] = Xtb.is_contiguous()
Ynb = m.rms_norm_forward(Xtb, eps)
Rnb = ref(Xtb.float(), eps)
dn = (Ynb.double()-Rnb.double()).abs()
out["nc_bf16_max_rel_err"] = (dn/Rnb.double().abs().clamp_min(1e-30)).max().item()
print(json.dumps(out))
