import torch, numpy as np, sys, json
sys.path.insert(0, "/root/cases/case_147")
import kernel

bits = np.arange(0, 65536, dtype=np.uint16)
vals = torch.from_numpy(bits.view(np.float16)).to(torch.float16)
finite = torch.isfinite(vals)
x = vals[finite].to("cuda")  # all finite fp16 values incl. signed zeros, subnormals, +/-65504
print("N =", x.numel())

out = kernel.run(x)
ref = kernel.reference(x.cpu()).to("cuda")  # FP64 atan of exact stored values
err = (out.double() - ref.double()).abs()
maxerr = err.max().item()
idx = int(err.argmax())
n_fail = int((err > 1e-6).sum())
print("max abs err:", maxerr)
print("argmax input:", x[idx].item(), "out:", out[idx].item(), "ref:", ref[idx].item())
print("num elements exceeding 1e-6:", n_fail)
if n_fail:
    bad = (err > 1e-6).nonzero().squeeze(1)[:10]
    for i in bad.tolist():
        print("bad:", x[i].item(), out[i].item(), ref[i].item(), err[i].item())
print(json.dumps({"metric":"max_abs_error_exhaustive_fp16","max_abs_err":maxerr,"n_fail":n_fail,"N":int(x.numel())}))