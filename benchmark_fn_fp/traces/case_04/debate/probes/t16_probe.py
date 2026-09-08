
import json, importlib.util, sys, traceback
out = {}
try:
    spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_04/kernel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
except Exception as e:
    print(json.dumps({"import_ok": False, "err": repr(e)})); sys.exit(0)

import torch
torch.manual_seed(2)
R, C, eps = 16, 128, 1e-6
x = torch.randn(R, C, device="cuda", dtype=torch.float32)
xb = x.to(torch.bfloat16)
try:
    y = m.rms_norm_forward(xb, eps)
    xf = xb.float().double()
    ref_fp32 = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps)   # fp32-style ref on bf16 values
    ref_liger = ref_fp32.to(torch.bfloat16).double()                       # Liger casts back to input dtype

    e1 = (y.double() - ref_fp32).abs()
    r1 = e1 / ref_fp32.abs().clamp_min(1e-30)
    e2 = (y.double() - ref_liger).abs()
    r2 = e2 / ref_liger.abs().clamp_min(1e-30)
    # if a checker compared in bf16 space
    y_as_bf16 = y.to(torch.bfloat16).double()
    e3 = (y_as_bf16 - ref_liger).abs()

    out.update({
        "ok": True,
        "returned_dtype": str(y.dtype),
        "input_dtype": str(xb.dtype),
        "max_abs_err_vs_fp32ref": e1.max().item(),
        "max_rel_err_vs_fp32ref": r1.max().item(),
        "max_abs_err_vs_ligerref_bf16": e2.max().item(),
        "max_rel_err_vs_ligerref_bf16": r2.max().item(),
        "max_abs_err_after_rounding_out_to_bf16": e3.max().item(),
        "exact_match_after_rounding": bool((y.to(torch.bfloat16) == ref_fp32.to(torch.bfloat16)).all().item()),
    })
except Exception as e:
    out["ok"] = False
    out["err"] = repr(e)
    out["tb"] = traceback.format_exc()[-1500:]
print(json.dumps(out))
