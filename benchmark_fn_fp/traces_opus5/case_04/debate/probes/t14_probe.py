
import json, importlib.util, sys, traceback
out = {}
try:
    spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_04/kernel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out["import_ok"] = True
    out["rsqrt_module"] = getattr(m.rsqrt, "__module__", str(m.rsqrt))
    out["rsqrt_repr"] = str(m.rsqrt)[:120]
except Exception as e:
    out["import_ok"] = False
    out["import_err"] = repr(e)
    print(json.dumps(out)); sys.exit(0)

import torch, triton
out["torch"] = torch.__version__
out["triton"] = triton.__version__
out["cuda_available"] = torch.cuda.is_available()

results = {}
try:
    torch.manual_seed(0)
    for (R, C) in [(8, 64), (4, 100), (3, 1), (17, 513)]:
        x = torch.randn(R, C, device="cuda", dtype=torch.float32)
        eps = 1e-6
        y = m.rms_norm_forward(x, eps)
        xd = x.double()
        ref = xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps)
        err = (y.double() - ref).abs()
        rel = err / ref.abs().clamp_min(1e-30)
        results[f"{R}x{C}"] = {
            "max_abs_err": err.max().item(),
            "max_rel_err": rel.max().item(),
            "out_dtype": str(y.dtype),
            "shape": list(y.shape),
            "all_finite": bool(torch.isfinite(y).all().item()),
        }
    out["launch_ok"] = True
except Exception as e:
    out["launch_ok"] = False
    out["launch_err"] = repr(e)
    out["tb"] = traceback.format_exc()[-1500:]
out["cases"] = results
print(json.dumps(out))
