import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel

results = {}
try:
    x = kernel.make_inputs()[0]
    out = kernel.run(x)
    ok = out.shape == x.shape and out.dtype == torch.int32 and out.is_cuda
    values_canonical = bool(((out >= 0) & (out < 3329)).all())
    results = {"compile_launch": True, "shape_dtype_device_ok": bool(ok),
               "values_canonical_0_3328": values_canonical,
               "min": int(out.min()), "max": int(out.max()),
               "exact_equal_make_inputs": bool(torch.equal(out, kernel.reference(x)))}
except Exception as e:
    results = {"compile_launch": False, "error": f"{type(e).__name__}: {e}"}
print(json.dumps(results))