import json, torch, sys
sys.path.insert(0, "/root/cases/case_152")
import kernel as K
res = {}
try:
    t = torch.zeros((7,64), dtype=torch.uint8, device="cuda")
    res["has_is_neg"] = hasattr(t, "is_neg")
    res["has_is_conj"] = hasattr(t, "is_conj")
    res["is_neg_value"] = t.is_neg()
    res["is_conj_value"] = t.is_conj()
    out = K.run(t)
    res["run_ok"] = True
    res["out_shape"] = list(out.shape)
    res["out_dtype"] = str(out.dtype)
    res["out_device"] = str(out.device)
except Exception as e:
    res["run_ok"] = False
    res["exception"] = repr(e)
print(json.dumps(res))