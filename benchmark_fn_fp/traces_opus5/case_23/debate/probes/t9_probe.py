
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_23/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
def trial(C, npix=64):
    torch.manual_seed(0)
    x = torch.randn(npix, C, device="cuda", dtype=torch.float32).contiguous()
    scale = torch.randn(C, device="cuda", dtype=torch.float32).contiguous()
    ref = x * scale
    try:
        out = m.scale_channels(x, scale, C)
        torch.cuda.synchronize()
        return {"status":"ran",
                "max_abs_err": (out-ref).abs().max().item(),
                "exact": bool(torch.equal(out, ref))}
    except Exception as e:
        return {"status":"error", "exc_type": type(e).__name__, "msg": str(e)[:300]}

for C in [512, 1024, 2048, 4096, 8192]:
    res[str(C)] = trial(C)

print(json.dumps({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "results": res}))
