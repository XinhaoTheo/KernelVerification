import torch, sys, traceback
sys.path.insert(0, "/root/cases/case_163")
import kernel
dev = "cuda"
torch.manual_seed(0)
v = torch.randn(4, 128, device=dev, dtype=torch.float16)
res = {"compiled": False, "exception": None, "max_abs_err": None}
try:
    out = kernel.run(v)
    torch.cuda.synchronize()
    res["compiled"] = True
    ref = torch.fft.rfft(v.to(torch.float64), dim=1)
    refc = torch.stack((ref.real, ref.imag), dim=-1)
    err = (out.double() - refc).abs()
    res["max_abs_err"] = err.max().item()
    res["shape"] = list(out.shape)
    res["dtype"] = str(out.dtype)
    res["finite"] = bool(torch.isfinite(out).all())
except Exception as e:
    res["exception"] = repr(e)[:800]
print(res)