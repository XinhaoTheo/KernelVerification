
import json, math, torch
import kernel as K

dev = "cuda" if torch.cuda.is_available() else "cpu"
# Full positive FP16 grid, finite, restricted to |x| where GELU error could matter (x <= 20)
bits = torch.arange(0, 65536, dtype=torch.int32)
allf16 = bits.to(torch.uint16).view(torch.float16)
pos = allf16[torch.isfinite(allf16) & (allf16 > 0)]
sel = pos[pos <= 20.0]
x = torch.cat([sel, -sel]).to(dev)
out = K.run(x)
ref = K.reference(x).to(torch.float64).to(dev)
err = (out.to(torch.float64) - ref).abs()
i = int(torch.argmax(err))

# Counterfactual: simulate kernel in FP32 math with exact A&S constant 0.3275911 (FP64 sim, no fp32 rounding) to isolate constant truncation
def sim(x64, c):
    m = x64.abs()
    t = 1.0 / (1.0 + c * m / math.sqrt(2.0))
    p = 1.06141 * t - 1.45315
    p = p * t + 1.42141
    p = p * t - 0.28450
    p = p * t + 0.25483
    p = p * t
    tail = 0.5 * p * torch.exp(-0.5 * m * m)
    prob = torch.where(x64 < 0, tail, 1.0 - tail)
    return x64 * prob

x64 = x.to(torch.float64)
ref_exact = sim(x64, 0.3275911)
ref_trunc = sim(x64, 0.3276)
print(json.dumps({
    "kernel_max_abs_err": float(err.max()),
    "tol": 1e-6,
    "kernel_violations": int((err > 1e-6).sum()),
    "worst_x": float(x.flatten()[i]),
    "sim_err_exact_const": float((ref_exact - ref).abs().max()),
    "sim_err_trunc_const": float((ref_trunc - ref).abs().max()),
    "sim_const_induced_extra": float((ref_trunc - ref_exact).abs().max()),
    "dev": dev,
}))
