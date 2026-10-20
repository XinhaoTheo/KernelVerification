
import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_99/kernel.py")
k = importlib.util.spec_from_file_location and importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)

# All finite FP16 bit patterns (includes signed zeros and subnormals)
bits = torch.arange(0, 65536, dtype=torch.int32)
b = bits.to(torch.uint16).view(torch.float16)
b64 = b.to(torch.float64)
finite = torch.isfinite(b64)
x_all = b[finite].clone()  # ~65504 values, includes -0 and +0
n = x_all.numel()
# Split into two chunks under 131072? no, n<=65504 fits
x = x_all.cuda()
out = k.run(x)
ref = torch.asinh(x.cpu().to(torch.float64))
err = (out.cpu().to(torch.float64) - ref).abs()
max_err = err.max().item()
worst_i = int(err.argmax())
res = {
    "metric": "max abs error over exhaustive finite FP16 domain",
    "n_tested": n,
    "max_abs_err": max_err,
    "tolerance": 1e-6,
    "passes": bool(max_err <= 1e-6),
    "worst_input": float(x[worst_i]),
    "worst_ref": float(ref[worst_i]),
    "worst_actual": float(out[worst_i]),
    "num_violations": int((err > 1e-6).sum()),
}
print(json.dumps(res))
