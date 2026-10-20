import torch, json

dev = 'cuda'
# All finite BF16 values in [-10,10]
bits = torch.arange(0, 65536, dtype=torch.int32)
all_bf16 = torch.tensor(bits.to(torch.uint16).numpy().tobytes(), dtype=torch.uint8).view(torch.bfloat16) if False else None
# build BF16 from raw uint16
u16 = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16)
b = torch.tensor(u16.numpy().view('uint16').copy(), dtype=torch.uint16)
bf = b.numpy().view(torch.dtype('torch.bfloat16')) if False else None
# simpler: use torch view trick
t = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16)
vals = t.view(torch.bfloat16)  # not supported directly; use frombuffer
import numpy as np
raw = np.arange(0, 65536, dtype=np.uint16)
x_all = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
finite = torch.isfinite(x_all)
inrange = (x_all.abs() <= 10) & finite
x = x_all[inrange].to(dev)
print("num legal values:", x.numel())

# fraction per kernel scheme
x32 = x.to(torch.float32)
integ = torch.floor(x32 + 0.5).to(torch.int32)
frac = x32 - integ.to(torch.float32)

# pure polynomial in FP64 (isolates approximation error from FP32 rounding)
f = frac.to(torch.float64)
p = torch.full_like(f, 0.0013391)
p = p * f + 0.009676
p = p * f + 0.055504
p = p * f + 0.24022
p = p * f + 0.69315
p = p * f + 1.0
scale = torch.exp2(integ.to(torch.float64))
exact = torch.exp2(f)

rel = ((p * scale) - exact).abs() / exact
i = rel.argmax()
print("poly-only max rel err:", rel[i].item(), "at x =", x[i].item(), "frac =", f[i].item())
print(json.dumps({"probe": "c1_poly_only_fp64", "num_legal_bf16": int(x.numel()),
                 "max_rel_err_poly_fp64": float(rel[i].item()),
                 "worst_x": float(x[i].item()),
                 "count_exceed_1e-6": int((rel > 1e-6).sum().item())}))
