import sys, json, torch
sys.path.insert(0, "/root/cases/case_126")
import kernel

bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
finite = torch.isfinite(bits)
x = bits[finite]  # all finite FP16 values incl signed zeros, subnormals, extrema
x = x.cuda()
out = kernel.run(x)
ref = torch.erf(x.to(torch.float64)).to(torch.float32)
err = (out.to(torch.float64) - ref.to(torch.float64)).abs()
max_err = err.max().item()
over = (err > 1e-6)
print(json.dumps({
    "metric": "max abs error vs FP64-erf rounded to FP32, exhaustive over all finite FP16 bit patterns",
    "n_values": int(x.numel()),
    "max_abs_err": max_err,
    "tolerance": 1e-6,
    "n_over_tolerance": int(over.sum().item()),
    "worst_x": float(x[err.argmax()].item()),
    "finite_out": bool(torch.isfinite(out).all().item()),
    "dtype": str(out.dtype),
}))
