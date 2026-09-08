
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev = "cuda"
rows, cols, eps = 8, 512, 1e-6

X32 = torch.randn(rows, cols, device=dev, dtype=torch.float32).contiguous()
Xb  = X32.to(torch.bfloat16).contiguous()

def ref(X, eps):
    # Liger behavior: fp32 accumulation, store Y in the INPUT storage dtype
    x = X.float()
    ms = (x*x).sum(-1, keepdim=True)/x.shape[-1]
    y = x * torch.rsqrt(ms + eps)
    return y.to(X.dtype)

res = {}

# --- bf16 storage path ---
out_b   = m.rms_norm_forward(Xb, eps)
ref_b   = ref(Xb, eps)            # bf16-rounded output (Liger-like)
ref_b32 = ref(Xb.float(), eps)    # fp32 output, same numeric values
res["bf16_in_out_dtype"] = str(out_b.dtype)
res["bf16_ref_dtype"]    = str(ref_b.dtype)
res["dtype_mismatch_bf16"] = str(out_b.dtype) != str(ref_b.dtype)
d_b = (out_b.float() - ref_b.float()).abs()
res["max_abs_err_bf16_vs_bf16ref"] = float(d_b.max())
res["max_rel_err_bf16_vs_bf16ref"] = float((d_b / ref_b.float().abs().clamp_min(1e-30)).max())
res["max_abs_err_bf16_vs_fp32ref"] = float((out_b - ref_b32).abs().max())
res["bf16_half_ulp_rel_approx"] = 2**-9
res["allclose_bf16_vals_rtol1e-2"] = bool(torch.allclose(out_b.float(), ref_b.float(), rtol=1e-2, atol=1e-2))
res["allclose_bf16_vals_rtol1e-5"] = bool(torch.allclose(out_b.float(), ref_b.float(), rtol=1e-5, atol=1e-6))
res["n_elems_differing_bf16"] = int((out_b.float() != ref_b.float()).sum())
res["n_elems_total"] = int(out_b.numel())

# --- fp32 contiguous control ---
out_f = m.rms_norm_forward(X32, eps)
ref_f = ref(X32, eps)
d_f = (out_f - ref_f).abs()
res["fp32_in_out_dtype"] = str(out_f.dtype)
res["max_abs_err_fp32"] = float(d_f.max())
res["max_rel_err_fp32"] = float((d_f / ref_f.abs().clamp_min(1e-30)).max())

# --- near-zero-magnitude rows (problem.txt calls these out) ---
Z32 = torch.zeros(4, cols, device=dev, dtype=torch.float32)
Z32[1] = 1e-20; Z32[2] = 1e-8; Z32[3] = torch.randn(cols, device=dev)*1e-6
o_z = m.rms_norm_forward(Z32, eps); r_z = ref(Z32, eps)
res["max_abs_err_fp32_nearzero"] = float((o_z - r_z).abs().max())
res["nearzero_out_finite"] = bool(torch.isfinite(o_z).all())
res["nearzero_row0_out_max"] = float(o_z[0].abs().max())
res["nearzero_row2_out_max"] = float(o_z[2].abs().max())

Zb = Z32.to(torch.bfloat16)
o_zb = m.rms_norm_forward(Zb, eps); r_zb = ref(Zb, eps)
res["nearzero_bf16_out_dtype"] = str(o_zb.dtype)
res["max_abs_err_bf16_nearzero_vs_bf16ref"] = float((o_zb.float() - r_zb.float()).abs().max())
res["nearzero_bf16_out_finite"] = bool(torch.isfinite(o_zb).all())

print(json.dumps(res))
