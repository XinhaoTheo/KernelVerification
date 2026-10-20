import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_100")
import kernel

# all finite fp16 values
bits = np.arange(65536, dtype=np.uint16)
vals = bits.view(np.float16)
finite = np.isfinite(vals)
xs = torch.from_numpy(vals[finite].copy()).to("cuda")  # signed zero included
N = xs.numel()
print("num finite fp16 values:", N)

# run in chunks within kernel launch limits (N<=131072, fine in one call)
out = kernel.run(xs)
ref = torch.asinh(xs.to(torch.float64)).to(torch.float32)
err = (out.to(torch.float64) - torch.asinh(xs.to(torch.float64))).abs()
max_err = err.max().item()
bad = (err > 1e-6).sum().item()
worst_i = int(err.argmax())
print("max_abs_err", max_err)
print("num_over_tol", bad)
print("worst x", xs[worst_i].item(), "out", out[worst_i].item(), "ref", torch.asinh(xs[worst_i].to(torch.float64)).item())
json.dump({"N": int(N), "max_abs_err": max_err, "num_over_tol": int(bad), "tol": 1e-6}, open("/tmp/res.json","w"))
