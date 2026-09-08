
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = {}
for name, shape in [("rank2", (4,128)), ("rank3", (2,4,128)), ("rank1", (128,))]:
    X = torch.randn(*shape, device="cuda")
    try:
        Y = m.rms_norm_forward(X, 1e-6)
        out[name] = {"ok": True, "out_shape": list(Y.shape)}
    except Exception as e:
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
print(json.dumps(out))
