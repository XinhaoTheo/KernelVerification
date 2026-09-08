
import importlib.util, torch, json, traceback
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

out = {}
torch.manual_seed(0)
eps = 1e-6
X32 = torch.randn(8, 64, device='cuda', dtype=torch.float32)
Xbf = X32.to(torch.bfloat16)

Y = m.rms_norm_forward(Xbf, eps)
out["input_dtype"] = str(Xbf.dtype)
out["output_dtype"] = str(Y.dtype)
out["dtype_follows_input"] = (Y.dtype == Xbf.dtype)

# fp32 reference computed from the bf16 stored values (upstream: fp32 accum)
xf = Xbf.float()
ms = (xf*xf).sum(dim=1, keepdim=True)/xf.shape[1]
ref_fp32 = xf * torch.rsqrt(ms + eps)
ref_bf16 = ref_fp32.to(torch.bfloat16)     # upstream Liger stores Y in input dtype

d32 = (Y.double() - ref_fp32.double()).abs()
out["max_abs_err_vs_fp32ref"] = d32.max().item()
out["max_rel_err_vs_fp32ref"] = (d32/ref_fp32.double().abs()).max().item()

dbf = (Y.double() - ref_bf16.double()).abs()
out["max_abs_err_vs_bf16ref"] = dbf.max().item()
out["max_rel_err_vs_bf16ref"] = (dbf/ref_bf16.double().abs().clamp_min(1e-30)).max().item()
out["frac_elems_differ_from_bf16ref"] = (dbf > 0).float().mean().item()

try:
    torch.allclose(Y, ref_bf16)
    out["allclose_vs_bf16ref_raises"] = False
except Exception as e:
    out["allclose_vs_bf16ref_raises"] = True
    out["allclose_error"] = type(e).__name__ + ": " + str(e)[:200]

# does casting kernel output down to bf16 recover the bf16 reference exactly?
out["kernel_cast_to_bf16_matches_bf16ref"] = bool(torch.equal(Y.to(torch.bfloat16), ref_bf16))

# fp32 input path for contrast
Y32 = m.rms_norm_forward(X32, eps)
out["fp32_input_output_dtype"] = str(Y32.dtype)
print(json.dumps(out))
