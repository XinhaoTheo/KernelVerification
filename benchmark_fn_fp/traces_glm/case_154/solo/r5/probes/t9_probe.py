import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"

def check(num, den, label):
    out = k.run(num.clone().to(dev), den.clone().to(dev))
    ref = k.reference(num.cpu(), den.cpu()).to(dev)
    err = (out - ref).abs()
    tol = 1e-7 + 5e-7 * ref.abs()
    viol = err > tol
    ratio = err / tol
    worst = ratio.max().item()
    detail = {}
    if viol.any():
        i = int(ratio.argmax()); r, cidx = divmod(i, 2)
        detail = {"row": r, "comp": cidx, "num": num[r].tolist(), "den": den[r].tolist(),
                  "out": out[r].tolist(), "ref": ref[r].tolist(),
                  "err": err.flatten()[i].item(), "tol": tol.flatten()[i].item()}
    print(label, {"violations": int(viol.sum()), "components": num.numel(),
                  "worst_err_over_tol": worst, "detail": detail})
    return int(viol.sum())

g = torch.Generator().manual_seed(0)
def fp16_uniform(shape, lo, hi):
    return ((torch.rand(shape, generator=g) * (hi - lo) + lo)).clamp(-65504, 65504).to(torch.float16)

total = 0
# random sweeps at extreme magnitudes (size within domain)
for scale in (65504.0, 30000.0, 1.0, 1e-4, 6.1e-5):
    num = (torch.randn(131072, 2, generator=g) * scale).clamp(-65504, 65504).to(torch.float16)
    den = (torch.randn(131072, 2, generator=g) * scale).clamp(-65504, 65504).to(torch.float16)
    z = (den == 0).all(dim=1); den[z, 0] = 1
    total += check(num, den, f"random_scale_{scale}")

# extreme magnitude quotient: max numerator, subnormal-ish denominator
num = torch.full((1000, 2), 65504.0, dtype=torch.float16)
den = torch.full((1000, 2), 6.1e-5, dtype=torch.float16)
den[:, 1] = float(torch.tensor(5.96e-8, dtype=torch.float16))
total += check(num, den, "extreme_magnitude")

# subnormal numerators and denominators
vals_sub = torch.tensor([5.96e-8, 1.19e-7, 5.96e-7, 6.1e-5], dtype=torch.float16)
num = torch.stack([vals_sub.repeat(250), (-vals_sub).repeat(250)], dim=1)
den = torch.stack([vals_sub.flip(0).repeat(250), vals_sub.repeat(250)], dim=1)
total += check(num, den, "subnormals")

# grid of tricky values
vals = torch.tensor([0.0, -0.0, 6.1e-5, -6.1e-5, 1e-4, 0.001, -0.001, 1.0, -1.0, 1.5, -2.25,
                     100.0, -100.0, 1000.0, -1000.0, 30000.0, -30000.0, 65504.0, -65504.0,
                     float(torch.tensor(65504.0/3, dtype=torch.float16))], dtype=torch.float16)
A, B = torch.meshgrid(vals, vals, indexing="ij")
nrows = torch.stack([A.flatten(), B.flatten()], dim=1)
dens = torch.tensor([[65504.0, -65504.0], [1.0, 6.1e-5], [30000.0, 30000.0], [-1000.0, 0.5],
                     [6.1e-5, 6.1e-5], [65504.0, float(torch.tensor(65504.0/3, dtype=torch.float16))],
                     [1.0, 1.0], [1.0, -1.0]], dtype=torch.float16)
num = nrows.repeat(len(dens), 1)
den = dens.repeat(len(nrows), 1)
total += check(num, den, "grid")

# make_inputs example and N=1
n, d = k.make_inputs()
total += check(n, d, "make_inputs_example")
total += check(torch.tensor([[1.0, -2.0]], dtype=torch.float16),
               torch.tensor([[3.0, 4.0]], dtype=torch.float16), "single_row")

print("TOTAL_VIOLATIONS", total)
import json
print(json.dumps({"total_violations": total}))
