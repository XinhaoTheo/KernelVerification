
import torch, sys
sys.path.insert(0, "/root/cases/case_156")
import kernel as K

def check(a, m, r, z, tag):
    out = K.run(a, m, r, z)
    ref = K.reference(a, m, r, z)
    ok = torch.equal(out, ref)
    if not ok:
        idx = (out != ref).nonzero()[:10]
        print("MISMATCH", tag, idx.tolist(), out[out!=ref][:10].tolist(), ref[ref!=out][:10].tolist())
    return ok

dev = "cuda"
fails = 0

# random sweeps
for seed in range(8):
    a, m, r, z = K.make_inputs(device=dev, rows=37*(seed+1)%200, channels=(seed%257)+1, seed=seed)
    fails += 0 if check(a, m, r, z, f"rand{seed}") else 1

# adversarial: extremes of a, m, r, z
for am in [-(2**31), -1, 0, 1, 2**31-1]:
  for mm in [2**30, 2**31-1]:
    for rr in [0, 1, 15, 31]:
      for zz in [-128, 0, 127]:
        a = torch.full((3, 7), am, device=dev, dtype=torch.int32)
        m = torch.full((7,), mm, device=dev, dtype=torch.int32)
        r = torch.full((7,), rr, device=dev, dtype=torch.int32)
        z = torch.full((7,), zz, device=dev, dtype=torch.int32)
        fails += 0 if check(a, m, r, z, f"ext {am},{mm},{rr},{zz}") else 1

# halfway cases: need P + 2^(30+r) odd multiple? halfway when P/2^(31+r) = k + 0.5, i.e. P = (2k+1)*2^(30+r). Pick r=0, m odd.
# P = a*m; choose a=1, m=3: P=3, 2^30 bias, R = (3+2^30)>>31 = 0 for m small... halfways need large m.
# Use m = 2^31-1 (odd), a odd: a*m = odd. P/2^31 = halfway when odd -> (P + 2^30)>>31 = (odd + even)>>31... P odd means P = 2k+1, then (2k+1+2^30) is odd+2^30; floor((P+2^30)/2^31) with P odd: (P+2^30)/2^31 = k/2 + ... exact half-integers occur when P = (2t+1)*2^30. With r=0, need a*m = (2t+1)*2^30; m odd so a must be (2t+1)*2^30 / m -> choose m=2^30 (not odd), a odd: a*2^30 = odd*2^30, P/2^31 = odd/2 = half-integer. 
for aa in [1, 3, -1, -3, 5, -5, 2**31-1, -(2**31)]:
    a = torch.tensor([[aa, aa, aa]], device=dev, dtype=torch.int32)
    m = torch.tensor([2**30, 2**30+1, 2**31-1], device=dev, dtype=torch.int32)
    r = torch.tensor([0, 0, 0], device=dev, dtype=torch.int32)
    z = torch.tensor([0, -1, 127], device=dev, dtype=torch.int32)
    fails += 0 if check(a, m, r, z, f"halfway {aa}") else 1

# saturation boundary: R = ±128..±300 with z
for aa in [1, 2, -1, -2, 4, 100, -100]:
    a = torch.tensor([[aa, aa]], device=dev, dtype=torch.int32)
    m = torch.tensor([2**31-1, 2**31-1], device=dev, dtype=torch.int32)
    r = torch.tensor([21, 22], device=dev, dtype=torch.int32)  # scale so R lands near 128/255
    z = torch.tensor([-128, 127], device=dev, dtype=torch.int32)
    fails += 0 if check(a, m, r, z, f"sat {aa}") else 1

# full random big sweep
g = torch.Generator().manual_seed(123)
a = torch.randint(-(2**31), 2**31, (512, 257), generator=g, dtype=torch.int64).to(torch.int32).to(dev)
m = torch.randint(2**30, 2**31, (257,), generator=g, dtype=torch.int64).to(torch.int32).to(dev)
r = torch.randint(0, 32, (257,), generator=g, dtype=torch.int32).to(dev)
z = torch.randint(-128, 128, (257,), generator=g, dtype=torch.int32).to(dev)
fails += 0 if check(a, m, r, z, "big512x257") else 1

# 1x1
a = torch.tensor([[7]], device=dev, dtype=torch.int32)
m = torch.tensor([2**31-1], device=dev, dtype=torch.int32)
r = torch.tensor([5], device=dev, dtype=torch.int32)
z = torch.tensor([-3], device=dev, dtype=torch.int32)
fails += 0 if check(a, m, r, z, "1x1") else 1

# preserve inputs
a2 = a.clone(); m2=m.clone(); r2=r.clone(); z2=z.clone()
K.run(a, m, r, z)
pres = torch.equal(a,a2) and torch.equal(m,m2) and torch.equal(r,r2) and torch.equal(z,z2)
print(torch.__version__)
import json
print(json.dumps({"fails": fails, "inputs_preserved": pres}))
