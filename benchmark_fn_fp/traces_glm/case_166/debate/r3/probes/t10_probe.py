
import sys, json, torch
sys.path.insert(0, "/root/cases/case_166")
from kernel import run

# focus: mantissa split boundary (u near 1.65*2^k) and extremes (x near 65504, smallest u)
def err_of(vals):
    x = vals.to(torch.float16).to("cuda")
    out = run(x)
    ref = torch.log1p(x.cpu().to(torch.float64))
    return (out.cpu().to(torch.float64) - ref).abs()

cands = []
# boundary region: mantissa in [1.60, 1.70] across exponents of u
for k in range(-11, 16):
    base = (1.65 * (2.0 ** k))
    lo, hi = 1.60 * (2.0**k), 1.70 * (2.0**k)
    n = 300
    vals = torch.linspace(lo, hi, n)
    cands.append(vals)
# extremes
cands += [torch.tensor([65504.0, 65503.0, 65000.0, -0.99951171875, -0.999]),
          torch.linspace(0.99, 1.01, 400), torch.linspace(-0.99, -1.0+1e-3, 400)]
vals = torch.cat(cands)
vals = vals[torch.isfinite(vals) & (vals > -1.0)]
x = vals.to(torch.float16)
e = err_of(x)
maxerr = float(e.max())
i = int(e.argmax())
# also the largest-exponent path: x = 65504
e_ext = err_of(torch.tensor([65504.0]))
print(json.dumps({
    "metric": "max abs error near mantissa=1.65 split boundary and extreme exponents",
    "n_tested": int(x.numel()),
    "max_abs_err": maxerr,
    "tolerance": 1e-6,
    "exceeds_1e-6": bool(maxerr > 1e-6),
    "worst_x": float(x[i]),
    "worst_err_at_65504": float(e_ext.max()),
}))
