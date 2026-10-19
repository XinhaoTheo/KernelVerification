import torch, sys, os
sys.path.insert(0, os.environ.get("CASE_DIR", "/root/cases/case_154"))
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

dev = "cuda"
def check(num, den, label):
    out = k.run(num.clone(), den.clone())
    ref = k.reference(num.cpu(), den.cpu()).to(dev)
    err = (out - ref).abs()
    tol = 1e-7 + 5e-7 * ref.abs()
    bad = (err > tol) & ~(torch.isclose(out, ref) & (ref == 0))  # signed zeros equal handled by err==0 anyway
    viol = (err > tol)
    n = viol.sum().item()
    worst = (err / tol).max().item()
    idx = (err / tol).argmax().item() if viol.any() else -1
    detail = {}
    if viol.any():
        i = (err / tol).argmax().item()
        r, cidx = divmod(i, 2)
        detail = {"row": r, "comp": cidx, "num": num[r].tolist(), "den": den[r].tolist(),
                  "out": out[r].tolist(), "ref": ref[r].tolist(),
                  "err": err.flatten()[i].item(), "tol": tol.flatten()[i].item()}
    print(label, {"violations": n, "size": num.numel(), "worst_err_over_tol": worst, "detail": detail})
    return n

# 1) structured cancellation: a*c ≈ -b*d with large magnitudes (products non-representable in fp32)
g = torch.Generator().manual_seed(0)
def build(size, scale_hi):
    num = torch.empty((size,2)); den = torch.empty((size,2))
    for i in range(size):
        c = float(torch.empty((), dtype=torch.float16).uniform_(-scale_hi, scale_hi, generator=g))
        d = float(torch.empty((), dtype=torch.float16).uniform_(-scale_hi, scale_hi, generator=g))
        if c == 0 and d == 0: c = 1.0
        # choose a,b large with a*c + b*d ≈ 0 but not exactly equal products
        a = float(torch.empty((), dtype=torch.float16).uniform_(-scale_hi, scale_hi, generator=g))
        # solve b = -a*c/d rounded to fp16
        b = float(torch.tensor(-a*c/d, dtype=torch.float16))
        if a == 0: a = 1.0
        num[i] = torch.tensor([a, b]); den[i] = torch.tensor([c, d])
    return num.to(torch.float16), den.to(torch.float16)

total = 0
for scale in (65504.0, 1024.0, 1.0, 6.1e-5):  # include subnormal-scale denominators
    num, den = build(20000, scale if scale <= 65504 else 65504.0)
    if scale < 1.0:
        # small denominators: c,d tiny subnormals
        den = (torch.randn(20000,2, generator=g) * 6e-5).to(torch.float16)
        z = (den == 0).all(dim=1); den[z,0] = 6.1e-5
        num = (torch.randn(20000,2, generator=g) * scale).to(torch.float16)
    total += check(num.to(dev), den.to(dev), f"structured_scale_{scale}")

# 2) full random sweep at extreme magnitudes
for scale in (65504.0, 30000.0, 1.0):
    num = (torch.randn(200000, 2, generator=g) * scale).to(torch.float16).to(dev)
    den = (torch.randn(200000, 2, generator=g) * scale).to(torch.float16).to(dev)
    z = (den == 0).all(dim=1); den[z, 0] = 1
    total += check(num, den, f"random_scale_{scale}")

# 3) extreme quotient: numerator max, denominator smallest subnormal (target huge, relative tolerance)
num = torch.full((1000, 2), 65504.0, dtype=torch.float16).to(dev)
den = torch.full((1000, 2), 6.1e-5, dtype=torch.float16)
den[:, 1] = 5.96e-8  # smallest subnormal
den = den.to(dev)
total += check(num, den, "extreme_magnitude")

# 4) all-bit patterns brute force on a grid (sampled full fp16 range)
vals = torch.tensor([0.0, -0.0, 6.1e-5, -6.1e-5, 6.104e-5, 1e-4, 0.001, -0.001, 1.0, -1.0, 1.5, -2.25,
                     100.0, -100.0, 1000.0, -1000.0, 30000.0, -30000.0, 65504.0, -65504.0, 65504.0/3],
                   dtype=torch.float16)
A, B = torch.meshgrid(vals, vals, indexing="ij")
rows = torch.stack([A.flatten(), B.flatten()], dim=1)
# combine numerator set with a few denominator rows
dens = torch.tensor([[65504.0, -65504.0], [1.0, 6.1e-5], [30000.0, 30000.0], [-1000.0, 0.5],
                     [6.1e-5, 6.1e-5], [65504.0, 65504.0/3], [1.0, 1.0], [1.0, -1.0]], dtype=torch.float16)
allnum = rows.repeat(len(dens), 1)
allden = dens.repeat(len(rows), 1)
total += check(allnum.to(dev), allden.to(dev), "grid_patterns")

print("TOTAL_VIOLATIONS", total)
