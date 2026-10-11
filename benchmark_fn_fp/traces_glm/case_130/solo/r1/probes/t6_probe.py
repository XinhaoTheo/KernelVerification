import torch, numpy as np, sys
sys.path.insert(0, "/root/cases/case_130")
import kernel

# All 65536 FP16 bit patterns
bits = np.arange(65536, dtype=np.uint16)
h = torch.from_numpy(bits.view(np.float16))
finite = torch.isfinite(h)
x_all = h[finite]  # all finite FP16 values (~61440)
print("num finite values:", x_all.numel())

# split into legal-size chunks (N<=131072; fits in one)
xs = x_all.to("cuda")
out = kernel.run(xs)
ref64 = kernel.reference(x_all).to("cuda")
err = (out.double() - ref64).abs()
mx = err.max().item()
bad = (err > 1e-6).sum().item()
idx = int(err.argmax())
print("max_abs_err:", mx)
print("num_violations(>1e-6):", bad)
print("worst x:", float(xs[idx]), "kernel:", float(out[idx]), "ref:", float(ref64[idx]))
# also check finiteness of output
print("output finite:", bool(torch.isfinite(out).all().item()))
import json
print(json.dumps({"metric":"max abs error vs FP64 reference over all finite FP16 inputs","max_abs_err":mx,"violations":bad,"tolerance":1e-6}))
