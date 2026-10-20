import torch, json, sys
sys.path.insert(0, "/root/cases/case_99")
import kernel as K

# All representable FP16 values with 0 <= |x| <= 1.0
bits = torch.arange(0, 65536, dtype=torch.int32)
vals = bits.to(torch.uint16).view(torch.float16) if hasattr(torch, 'uint16') else torch.tensor(bits.numpy(), dtype=torch.float16)
# robust: use bit view
vals = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
vals = vals[torch.isfinite(vals)]
m = vals.abs()
sub = vals[(m > 0) & (m <= 1.0)]
x = sub.to(torch.float64)

def poly64(t):
    sq = t*t
    p = -0.0011264356
    p = p*sq + 0.0059712074
    p = p*sq - 0.015329496
    p = p*sq + 0.027586811
    p = p*sq - 0.043983627
    p = p*sq + 0.074917369
    p = p*sq - 0.16666223
    p = p*sq + 0.99999994
    return t*p

approx = poly64(m.to(torch.float64)[(m>0)&(m<=1.0)].to(torch.float64)) if False else None
mag = sub.abs().to(torch.float64)
approx = poly64(mag)
exact = torch.asinh(mag)
err = (approx - exact).abs()
i = err.argmax()
print(json.dumps({
    "metric": "intrinsic FP64 polynomial error over all representable FP16 |x| in (0,1]",
    "n_values": int(sub.numel()),
    "max_abs_err": float(err[i]),
    "tolerance": 1e-6,
    "worst_x": float(sub[i]),
    "exceeds_1e-6": bool((err > 1e-6).any()),
    "n_exceed": int((err > 1e-6).sum()),
}))