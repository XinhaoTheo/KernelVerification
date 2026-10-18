
import torch, math, json, sys
import numpy as np
sys.path.insert(0, "/root/cases/case_128")
import kernel

# all 65536 FP16 bit patterns via numpy uint16 -> float16
bits = np.arange(65536, dtype=np.uint16)
raw = bits.view(np.float16)
x = torch.from_numpy(raw.copy())
x = x[torch.isfinite(x)]
print("num finite fp16 values:", x.numel())

xg = x.to("cuda")
out = kernel.run(xg)
ref = kernel.reference(xg).to(torch.float32)
err = (out - ref).abs()
maxerr = err.max().item()
imax = int(err.argmax().item())
xv = x[imax].item()
viol = (err > 1e-6).sum().item()
print("max abs err:", maxerr)
print("argmax x:", xv, "actual:", out[imax].item(), "ref:", ref[imax].item())
print("num violating elements:", viol)
if viol:
    idxs = torch.nonzero(err > 1e-6).flatten()
    print("sample violating inputs:", x[idxs[:20]].tolist())
    # report max error among violators
    print("max violating err:", err[idxs].max().item())
# also check finiteness of output
print("output all finite:", bool(torch.isfinite(out).all().item()))
print(json.dumps({"metric": "max abs error over all finite FP16 values", "num_inputs": x.numel(), "max_abs_err": maxerr, "tolerance": 1e-6, "violations": int(viol), "x_at_max": xv, "actual_at_max": out[imax].item(), "ref_at_max": ref[imax].item()}))
