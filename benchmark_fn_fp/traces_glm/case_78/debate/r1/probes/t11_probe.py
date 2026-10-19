import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel
info = {}
try:
    x = torch.randint(0, 3329, (2, 64), dtype=torch.int32, device="cuda")
    info["has_is_neg"] = hasattr(x, "is_neg")
    info["has_is_conj"] = hasattr(x, "is_conj")
    info["is_neg_value"] = bool(x.is_neg()) if info["has_is_neg"] else None
    info["is_conj_value"] = bool(x.is_conj()) if info["has_is_conj"] else None
    out = kernel.run(x)
    info["run_succeeded"] = True
    info["output_shape"] = list(out.shape)
    info["output_dtype"] = str(out.dtype)
    info["x_unchanged"] = bool(((x >= 0) & (x < 3329)).all())
except Exception as e:
    info["run_succeeded"] = False
    info["exception_type"] = type(e).__name__
    info["exception_msg"] = str(e)[:300]
print(json.dumps({"metric": "run_availability_and_output_presence", **info}))