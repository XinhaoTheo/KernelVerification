import torch, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
tol_a, tol_r = 1e-7, 5e-7
fails = []

def check(name, num, den):
    num = num.to(dev); den = den.to(dev)
    out = kernel.run(num, den)
    ref = kernel.reference(num, den).cpu().double()
    outc = out.cpu().double()
    diff = (outc - ref).abs()
    allowed = tol_a + tol_r * ref.abs()
    bad = diff > allowed
    nonfinite = ~torch.isfinite(outc)
    nbad = int(bad.sum()); nnf = int(nonfinite.sum())
    if nbad or nnf:
        m = (bad | nonfinite).flatten()
        fails.append((name, nbad, nnf, outc.flatten()[m.nonzero()[:5]].tolist(), ref.flatten()[m.nonzero()[:5]].tolist()))
    return nbad, nnf, outc.numel()

tot = 0; badtot = 0
for seed, size in [(0, 4093), (1, 131072), (2, 131072)]:
    g = torch.Generator().manual_seed(seed)
    num = torch.randn((size,2), generator=g).to(torch.float16)
    den = torch.randn((size,2), generator=g).to(torch.float16)
    den[(den==0).all(dim=1), 0] = 1
    nb, nf, n = check(f"rand seed{seed} size{size}", num, den); badtot += nb+nf; tot += n

special = torch.tensor([0.0,-0.0, 65504.0,-65504.0, 6.103515625e-05,-6.103515625e-05,
                        5.960464477539063e-08,-5.960464477539063e-08, 1.0,-1.0, 2048.0, 0.001, 300.0], dtype=torch.float16)
s1, s2 = torch.meshgrid(special, special, indexing="ij")
num = torch.stack([s1.reshape(-1), s2.reshape(-1)], dim=1)
s3, s4 = torch.meshgrid(special, special, indexing="ij")
den = torch.stack([s3.reshape(-1), s4.reshape(-1)], dim=1)
keep = ~(den == 0).all(dim=1)
num = num[keep]; den = den[keep]
nb, nf, n = check("special grid", num, den); badtot += nb+nf; tot += n

for seed in range(5):
    g = torch.Generator().manual_seed(100+seed)
    size = 65536
    den = torch.randn((size,2), generator=g).to(torch.float16)
    k = torch.randn((size,1), generator=g).to(torch.float16)
    num = (den.double() * k.double()).to(torch.float16)
    keep = ~(den==0).all(dim=1)
    nb, nf, n = check(f"cancellation seed{seed}", num[keep], den[keep]); badtot += nb+nf; tot += n

g = torch.Generator().manual_seed(7)
size = 4096
num = torch.empty((size,2)); num.uniform_(-65504, 65504, generator=g)
den = (torch.rand((size,2), generator=g) * 1e-7).to(torch.float16)
zz = (den==0).all(dim=1); den[zz,0] = torch.tensor(6e-8, dtype=torch.float16)
num = num.to(torch.float16)
nb, nf, n = check("extreme ratio", num, den); badtot += nb+nf; tot += n

base = torch.randn((10000,2), dtype=torch.float16)
numv = base[500:9000]
denb = torch.randn((20000,2), dtype=torch.float16)
denv = denb[100:8500]
denv = denv.clone(); denv[(denv==0).all(dim=1), 0] = 1
nb, nf, n = check("storage offset views", numv, denv); badtot += nb+nf; tot += n

# shared backing storage
buf = torch.randn((2, 8192), dtype=torch.float16).cuda()
numv = buf[0]; denv = buf[1]
denv = denv.clone(); denv[(denv==0).all(dim=1), 0] = 1
numv_c = numv.clone()
before = numv_c.clone()
nb, nf, n = check("shared storage", numv, denv); badtot += nb+nf; tot += n
inputs_preserved = bool(torch.equal(numv, before))

import json
print(json.dumps({"total_components": tot, "bad": badtot, "inputs_preserved": inputs_preserved, "fail_cases": fails if fails else None}))
