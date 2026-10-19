import torch, numpy as np, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

bits = np.arange(0, 65536, dtype=np.uint16)
vals = torch.from_numpy(bits.view(np.float16)).to(torch.float16)
finite = torch.isfinite(vals)
x = vals[finite].to("cuda")  # all finite fp16 values, includes subnormals, signed zeros, +/-65504
print("N =", x.numel())

out = kernel.run(x)
ref = kernel.reference(x.cpu().to(torch.float64)).to("cuda")
err = (out.double() - ref).abs()
maxerr = err.max().item()
idx = int(err.argmax())
print("max abs err:", maxerr)
print("argmax input:", x[idx].item(), "out:", out[idx].item(), "ref:", ref[idx].item())
n_fail = int((err > 1e-6).sum())
print("num elements exceeding 1e-6:", n_fail)
if n_fail:
    bad = (err > 1e-6).nonzero().squeeze(1)[:10]
    for i in bad.tolist():
        print("bad:", x[i].item(), out[i].item(), ref[i].item(), err[i].item())
import json
print(json.dumps({"metric":"max_abs_error_exhaustive_fp16","max_abs_err":maxerr,"n_fail":n_fail,"N":int(x.numel())}))
