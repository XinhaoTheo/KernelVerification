
import json, traceback, torch, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_23/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
def trial(C):
    torch.manual_seed(0)
    N,H,W = 2,4,5
    # honor kernel's own NHWC convention: channels contiguous, last axis
    x = torch.randn(N,H,W,C, device="cuda", dtype=torch.float32).contiguous()
    scale = torch.randn(C, device="cuda", dtype=torch.float32).contiguous()
    ref = x * scale  # broadcast over last (channel) axis
    try:
        out = m.scale_channels(x, scale, C)
        torch.cuda.synchronize()
        err = (out - ref).abs().max().item()
        return {"status":"ran", "max_abs_err": err, "exact": bool(torch.equal(out, ref))}
    except Exception as e:
        return {"status":"error", "exc_type": type(e).__name__,
                "msg": str(e)[:300]}

for C in [4, 64, 3, 96, 192, 320, 5]:
    res[str(C)] = trial(C)

try:
    import triton
    tv = triton.__version__
except Exception:
    tv = None
summary = {"triton_version": tv, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "results": res}
print(json.dumps(summary))
