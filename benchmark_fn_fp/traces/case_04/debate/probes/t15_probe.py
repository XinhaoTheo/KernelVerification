
import json, importlib.util, sys, traceback, math
out = {}
try:
    spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_04/kernel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
except Exception as e:
    print(json.dumps({"import_ok": False, "err": repr(e)})); sys.exit(0)

import torch
torch.manual_seed(1)
C = 64
scales = [1.0, 1e-4, 1e-8, 1e-20, 1e-24, 0.0]
base = torch.randn(len(scales), C, device="cuda", dtype=torch.float32)
x = torch.stack([base[i] * s for i, s in enumerate(scales)]).contiguous()

res = {}
try:
    for eps in [1e-6, 1e-5]:
        y = m.rms_norm_forward(x, eps)
        xd = x.double()
        ms = (xd * xd).mean(-1, keepdim=True)
        ref = xd * torch.rsqrt(ms + eps)
        err = (y.double() - ref).abs()
        rel = err / ref.abs().clamp_min(1e-300)
        per_row = []
        for i, s in enumerate(scales):
            ms32 = (x[i].double() * x[i].double()).sum().item() / C
            # fp32 accumulation of x*x for this row
            ms32_f32 = (x[i] * x[i]).sum().item() / C
            per_row.append({
                "scale": s,
                "mean_square_fp64": ms32,
                "mean_square_fp32_torch": ms32_f32,
                "max_abs_err": err[i].max().item(),
                "max_rel_err": rel[i].max().item(),
                "ref_finite": bool(torch.isfinite(ref[i]).all().item()),
                "out_finite": bool(torch.isfinite(y[i]).all().item()),
                "implied_rstd_kernel": (y[i, 0] / x[i, 0]).item() if x[i, 0].item() != 0 else None,
                "expected_rstd": (1.0 / math.sqrt(ms32 + eps)),
            })
        res[f"eps={eps}"] = {
            "global_max_abs_err": err.max().item(),
            "global_max_rel_err": rel.max().item(),
            "zero_row_out_max_abs": y[-1].abs().max().item(),
            "rows": per_row,
        }
    out["ok"] = True
except Exception as e:
    out["ok"] = False
    out["err"] = repr(e)
    out["tb"] = traceback.format_exc()[-1500:]
out["results"] = res
print(json.dumps(out))
