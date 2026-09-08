import json, importlib.util, torch, traceback
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_04/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}
eps = 1e-6
for name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
    x = torch.randn(128, 256, device="cuda").to(dt)
    out = m.rms_norm_forward(x, eps)
    ref = (x.float() * torch.rsqrt((x.float()**2).mean(-1, keepdim=True) + eps)).to(dt)
    entry = {"in_dtype": str(dt), "out_dtype": str(out.dtype),
             "dtype_matches_input": out.dtype == dt,
             "out_shape": list(out.shape)}
    # dtype-strict comparison
    try:
        torch.testing.assert_close(out, ref)
        entry["assert_close_strict"] = "passed"
    except Exception as e:
        entry["assert_close_strict"] = type(e).__name__ + ": " + str(e).strip().splitlines()[0][:200]
    # allclose with dtype-mismatched expected
    try:
        entry["allclose_raw"] = bool(torch.allclose(out, ref, rtol=1e-2, atol=1e-2))
    except Exception as e:
        entry["allclose_raw"] = "raised: " + type(e).__name__ + ": " + str(e)[:150]
    res[name] = entry
print(json.dumps(res))
