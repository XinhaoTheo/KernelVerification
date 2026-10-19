import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_04/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
eps = 1e-6
n_rows, n_cols = 4, 512
# bf16 round-trip of extremely tiny rows: 1e-20 and 1e-22 (x*x underflows fp32 to 0)
rows = []
for scale in [1e-19, 1e-20, 1e-22, 1e-24]:
    x = (torch.randn(n_cols, device=dev) * scale).to(torch.bfloat16)
    rows.append(x.float())
X = torch.cat(rows).reshape(len(rows), n_cols).contiguous()
Y = kern.rms_norm_forward(X, eps)
# fp32-accumulated torch reference (same underflow)
X32 = X.float(); ms32 = (X32*X32).sum(dim=1)/n_cols
rstd32 = 1.0/torch.sqrt(ms32+eps); Y32 = X32*rstd32[:,None]
# fp64 higher-precision reference
X64 = X.double(); ms64 = (X64*X64).sum(dim=1)/n_cols
rstd64 = 1.0/torch.sqrt(ms64+eps); Y64 = X64*rstd64[:,None]
err_vs_fp32 = (Y-Y32).abs().max().item()
err_vs_fp64 = (Y.double()-Y64).abs().max().item()
rel_vs_fp64 = ((Y.double()-Y64).abs()/Y64.abs()).max().item()
print(json.dumps({"scales": [1e-19,1e-20,1e-22,1e-24],
 "mean_square_fp32": ms32.tolist(), "mean_square_fp64": ms64.tolist(),
 "kernel_ms_underflowed_rows": int((ms32==0).sum()), "eps": eps,
 "max_abs_err_vs_fp32_ref": err_vs_fp32,
 "max_abs_err_vs_fp64_ref": err_vs_fp64,
 "max_rel_err_vs_fp64_ref": rel_vs_fp64,
 "allclose_vs_fp32_ref": bool(torch.allclose(Y, Y32, rtol=1e-5, atol=1e-30)),
 "device": dev}))