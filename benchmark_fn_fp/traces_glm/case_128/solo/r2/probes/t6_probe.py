
import torch, math, json, sys
sys.path.insert(0, "/root/cases/case_128")
import kernel

# all finite FP16 values
bits = torch.arange(65536, dtype=torch.int32)
u16 = torch.tensor(bits.numpy().tobytes(), dtype=torch.uint8).numpy()
import numpy as np
raw = np.frombuffer(bits.numpy().astype(np.uint16).tobytes(), dtype=np.float16)
x = torch.from_numpy(raw.copy())
finite = torch.isfinite(x)
x = x[finite]  # drop inf/nan bit patterns
print("num finite fp16 values:", x.numel())

xg = x.to("cuda")
out = kernel.run(xg)
ref = kernel.reference(xg).to(torch.float32)
err = (out - ref).abs()
maxerr = err.max().item()
imax = int(err.argmax().item())
xv = x[imax].item()
print("max abs err:", maxerr)
print("argmax x:", xv, "actual:", out[imax].item(), "ref:", ref[imax].item())
viol = (err > 1e-6).sum().item()
print("num violating elements:", viol)
if viol:
    idxs = torch.nonzero(err > 1e-6).flatten()
    vxs = x[idxs[:20]].tolist()
    print("sample violating inputs:", vxs)
print(json.dumps({"metric":"max abs error over all finite FP16","max_abs_err":maxerr,"violations":int(viol),"x_at_max":xv,"actual_at_max":out[imax].item(),"ref_at_max":ref[imax].item()}))
