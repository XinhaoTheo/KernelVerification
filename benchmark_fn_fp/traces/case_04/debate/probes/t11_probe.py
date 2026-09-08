
import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(2)
eps = 1e-6
res = {"eps": eps, "cases": {}}
EPS32 = 1.1920929e-7  # 2^-23, 1 fp32 ULP relative step

for (R, C, scale, tag) in [(16, 1024, 1.0, "randn_1024"),
                           (16, 64, 1.0, "randn_64"),
                           (8, 4096, 1e3, "randn_4096_scale1e3"),
                           (8, 128, 1e-2, "randn_128_scale1e-2")]:
    X = (torch.randn(R, C, device='cuda', dtype=torch.float32) * scale).contiguous()
    Y = m.rms_norm_forward(X, eps)
    Xd = X.double()
    msd = (Xd*Xd).sum(dim=1, keepdim=True)/C
    rstd_ref = 1.0/torch.sqrt(msd + eps)
    Yref = Xd * rstd_ref
    # least-squares implied rstd per row (fp64), averages out elementwise fp32 rounding
    implied = ((Y.double()*Xd).sum(dim=1, keepdim=True)/(Xd*Xd).sum(dim=1, keepdim=True))
    rstd_rel = ((implied - rstd_ref).abs()/rstd_ref.abs())
    yrel = ((Y.double()-Yref).abs()/Yref.abs().clamp_min(1e-300))
    # torch fp32 reference (divide+sqrt in fp32)
    ms32 = (X*X).sum(dim=1, keepdim=True)/C
    ytorch = X * (1.0/torch.sqrt(ms32 + eps))
    yrel32 = ((Y.double()-ytorch.double()).abs()/ytorch.double().abs().clamp_min(1e-300))
    res["cases"][tag] = {
        "shape": [R, C],
        "max_rstd_rel_err": rstd_rel.max().item(),
        "max_rstd_rel_err_in_ulps": rstd_rel.max().item()/EPS32,
        "max_y_rel_err_vs_fp64": yrel.max().item(),
        "max_y_rel_err_vs_fp64_in_ulps": yrel.max().item()/EPS32,
        "max_y_rel_err_vs_torch_fp32": yrel32.max().item(),
        "max_y_rel_err_vs_torch_fp32_in_ulps": yrel32.max().item()/EPS32,
        "allclose_torch_fp32_rtol1e-5": bool(torch.allclose(Y, ytorch, rtol=1e-5, atol=1e-6)),
        "allclose_torch_fp32_rtol1e-6": bool(torch.allclose(Y, ytorch, rtol=1e-6, atol=0.0)),
    }
print(json.dumps(res))
