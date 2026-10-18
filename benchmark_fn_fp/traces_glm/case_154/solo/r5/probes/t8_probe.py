import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"

def check(num, den, label):
    out = k.run(num.clone(), den.clone())
    ref = k.reference(num.cpu(), den.cpu()).to(dev)
    err = (out - ref).abs()
    tol = 1e-7 + 5e-7 * ref.abs()
    viol = err > tol
    n = int(viol.sum()); total = num.numel()
    ratio = (err / tol)
    worst = ratio.max().item() if torch.isfinite(ratio).all() else float('nan')
    detail = {}
    if viol.any():
        i = int(ratio.argmax())
        r, cidx = divmod(i, 2)
        detail = {"row": r, "comp": cidx, "num": num[r].tolist(), "den": den[r].tolist(),
                  "out": out[r].tolist(), "ref": ref[r].tolist(),
                  "err": err.flatten()[i].item(), "tol": tol.flatten()[i].item()}
    print(label, {"violations": n, "components": total, "worst_err_over_tol": worst, "detail": detail})
    return n

g = torch.Generator().manual_seed(0)
def fp16_uniform(shape, lo, hi):
    t = (torch.rand(shape, generator=g) * (hi - lo) + lo)
    return t.clamp(-65504.0, 65504.0).to(torch.float16)

total = 0
# structured catastrophic cancellation with large magnitudes
for scale in (65504.0, 1024.0, 1.0):
    size = 20000
    c = fp16_uniform((size,), -scale, scale).to(torch.float64)
    d = fp16_uniform((size,), -scale, scale).to(torch.float64)
    d[(d == 0)] = 1.0
    a = fp16_uniform((size,), -scale, scale).to(torch.float64)
    a[(a == 0)] = 1.0
    b = (-a * c / d).clamp(-65504, 65504).to(torch.float16).to(torch.float64)
    num = torch.stack([a.to(torch.float16), b.to(torch.float16)], dim=1)
    den = torch.stack([c.to(torch.float16), d.to(torch.float16)], dim=1)
    z = (den == 0).all(dim=1)
    if z.any(): den[z, 0] = torch.tensor(1.0, dtype=torch.float16)
    total += check(num.to(dev), den.to(dev), f"structured_cancel_scale_{scale}")

# random sweeps at extreme magnitudes (clamped finite)
for scale in (65504.0, 30000.0, 1.0, 1e-4, 6.1e-5):
    num = (torch.randn(200000, 2, generator=g) * scale).clamp(-65504, 65504).to(torch.float16)
    den = (torch.randn(200000, 2, generator=g) * scale).clamp(-65504, 65504).to(torch.float16)
    z = (den == 0).all(dim=1); den[z, 0] = 1
    total += check(num.to(dev), den.to(dev), f"random_scale_{scale}")

# grid of tricky values
vals = torch.tensor([0.0, -0.0, 6.1e-5, -6.1e-5, 1e-4, 0.001, -0.001, 1.0, -1.0, 1.5, -2.25,
                     100.0, -100.0, 1000.0, -1000.0, 30000.0, -30000.0, 65504.0, -65504.0,
                     float(torch.tensor(65504.0/3, dtype=torch.float16))], dtype=torch.float16)
A, B = torch.meshgrid(vals, vals, indexing="ij")
nrows = torch.stack([A.flatten(), B.flatten()], dim=1)
dens = torch.tensor([[65504.0, -65504.0], [1.0, 6.1e-5], [30000.0, 30000.0], [-1000.0, 0.5],
                     [6.1e-5, 6.1e-5], [65504.0, 65504.0/3], [1.0, 1.0], [1.0, -1.0]], dtype=torch.float16)
num = nrows.repeat(len(dens), 1).to(dev)
den = dens.repeat(len(nrows), 1).to(dev)
total += check(num, den, "grid")

n, d = k.make_inputs()
total += check(n, d, "make_inputs_example")
print("TOTAL_VIOLATIONS", total)
