import json, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel, torch
m, b, i = kernel.make_inputs(device="cuda", batch=7, length=32, seed=0)
try:
    out = kernel.run(m, b, i)
    ref = kernel.reference(m, b, i)
    exact = bool((out == ref).all().item())
    torch.cuda.synchronize()
    print(json.dumps({"claim": "c1", "compiled_ran": True, "exact_match_default": exact, "max_abs_err": int((out - ref).abs().max().item()) if not exact else 0}))
except Exception as e:
    print(json.dumps({"claim": "c1", "compiled_ran": False, "error": f"{type(e).__name__}: {e}"}))