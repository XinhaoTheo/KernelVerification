import torch, sys, os
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
fails = []
tol_a, tol_r = 1e-7, 5e-7

def check(name, num, den):
    num = num.to(dev); den = den.to(dev)
    out = kernel.run(num, den)
    ref = kernel.reference(num, den)  # FP64 on gpu -> fine
    ref = ref.cpu().double(); out = out.cpu().double()
    diff = (out - ref).abs()
    allowed = tol_a + tol_r * ref.abs()
    bad = diff > allowed
    nonfinite = ~torch.isfinite(out)
    nbad = int(bad.sum()); nnf = int(nonfinite.sum())
    if nbad or nnf:
        idx = (bad | nonfinite).nonzero()[:5]
        fails.append((name, nbad, nnf, out.flatten()[(bad|nonfinite).flatten().nonzero()[:5]].tolist(), ref.flatten()[(bad|nonfinite).flatten()[:5]].tolist()))
    return nbad, nnf, out.numel()

tot = 0; badtot = 0
# 1. random sweeps
for seed, size in [(0, 4093), (1, 131072), (2, 131072)]:
    g = torch.Generator().manual_seed(seed)
    num = torch.randn((size,2), generator=g).to(torch.float16)
    den = torch.randn((size,2), generator=g).to(torch.float16)
    den[(den==0).all(dim=1), 0] = 1
    nb, nf, n = check(f"rand seed{seed} size{size}", num, den); badtot += nb+nf; tot += n

# 2. grid of special values incl. extrema, subnormals, signed zeros
special = torch.tensor([0.0,-0.0, 65504.0, -65504.0, 6.103515625e-05, -6.103515625e-05,
                       5.960464477539063e-08, -5.960464477539063e-08, 1.0, -1.0, 2048.0, 0.001, 300.0], dtype=torch.float16)
m = len(special)
a, b = torch.meshgrid(special, special, indexing="ij")
num = torch.stack([a.reshape(-1), b.reshape(-1)], dim=1)
c, d = torch.meshgrid(special, special, indexing="ij")
den = torch.stack([c.reshape(-1), d.reshape(-1)], dim=1)
zz = (den == 0).all(dim=1)
den[zz, 0] = 1.0  # contract forbids both zero
num = num[~zz]; den = den[~zz]
nb, nf, n = check("special grid", num, den); badtot += nb+nf; tot += n

# 3. random with cancellation bias: numerator proportional to denominator
for seed in range(5):
    g = torch.Generator().manual_seed(100+seed)
    size = 65536
    den = torch.randn((size,2), generator=g).to(torch.float16)
    k = torch.randn((size,1), generator=g).to(torch.float16)
    num = (den.double() * k.double()).to(torch.float16)  # nearly parallel -> small imag component
    zz = (den==0).all(dim=1); den[zz,0]=1.0; num=num[~zz]; den=den[~zz]
    nb, nf, n = check(f"cancellation seed{seed}", num, den); badtot += nb+nf; tot += n

# 4. extreme scale ratios: numerator huge, denominator tiny subnormal
g = torch.Generator().manual_seed(7)
size = 4096
num = torch.empty((size,2)); num.uniform_(-65504, 65504, generator=g)
den = (torch.rand((size,2), generator=g) * 1e-7).to(torch.float16)  # subnormal region
den[(den==0).all(dim=1)] = torch.tensor([6e-8], dtype=torch.float16).item()
num = num.to(torch.float16)
nb, nf, n = check("extreme ratio", num, den); badtot += nb+nf; tot += n

# 5. contiguous views with nonzero storage offset, shared storage
base = torch.randn((10000, 2), dtype=torch.float16)
numv = base[500:9000]
denb = torch.randn((20000, 2), dtype=torch.float16)
denv = denb[100:8100]
denv[(denv==0).all(dim=1), 0] = 1
nb, nf, n = check("storage offset views", numv, denv); badtot += nb+nf; tot += n

print({"total_components": tot, "bad": badtot, "fail_cases": fails if fails else None})
