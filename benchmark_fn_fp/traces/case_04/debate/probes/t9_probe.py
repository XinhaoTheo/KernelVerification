
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res = {}
for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]:
    X = torch.randn(4, 512, device="cuda").to(dt)
    try:
        Y = m.rms_norm_forward(X, 1e-6)
        res[name] = {"in_dtype": str(X.dtype), "out_dtype": str(Y.dtype),
                     "matches_input_dtype": Y.dtype == X.dtype, "shape": list(Y.shape)}
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(res))
