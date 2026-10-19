
import torch, json, sys
sys.path.insert(0, "/root/cases/case_157")
import kernel

dev = "cuda"
# search legal inputs where fp64 path differs from exact int64 single-rounding
found = []
m = ((1<<31) - 1)  # max multiplier
r = 0
rng = torch.Generator().manual_seed(0)

def exact(a, m, r, z=0):
    P = a * m
    R = (P + (1 << (30 + r))) >> (31 + r)
    return max(-128, min(127, R + z))

# candidates: a such that a*m is odd*... halfway means P + 2^(30+r) = k*2^(31+r) + 2^(31+r) i.e.
# actual halfway: a*m/2^(31+r) = t + 0.5  -> a*m = (2t+1)*2^(30+r). a odd, m odd gives odd product -> not halfway.
# mismatch more likely when |a*m| > 2^53 so fp64 product rounds.
# test a bunch of large a values
for _ in range(40):
    a = int(torch.randint(1<<30, 1<<31, (1,), generator=rng).item())
    for sign in (1, -1):
        av = sign * a
        # check fp64 emulation vs exact
        pf = float(av) * float(m)  # fp64 rounding of conversion+product
        scaled = pf / float(1 << 31)
        import math
        kf = math.floor(scaled + 0.5)
        re = exact(av, m, r)
        if kf != re:
            found.append((av, m, r, kf, re))
print("emulation mismatches:", len(found), found[:3])

# Now run actual kernel on adversarial tensor
vals = []
if found:
    for av, m_, r_, kf, re in found[:5]:
        vals.append(av)
else:
    vals = [2147483647, -2147483648, 1234567891, -1234567891]
acc = torch.tensor(vals, dtype=torch.int32, device=dev).view(len(vals),1)
mult = torch.full((1,), m, dtype=torch.int32, device=dev)
sh = torch.zeros(1, dtype=torch.int32, device=dev)
zp = torch.zeros(1, dtype=torch.int32, device=dev)
out = kernel.run(acc, mult, sh, zp).cpu()
ref = kernel.reference(acc, mult, sh, zp).cpu()
print("kernel:", out.flatten().tolist())
print("ref   :", ref.flatten().tolist())
print("mismatch:", not torch.equal(out, ref))

# broad randomized check including r in 0..31
g = torch.Generator().manual_seed(1)
a = torch.randint(-(1<<31), 1<<31, (512, 17), generator=g, dtype=torch.int64).to(torch.int32).to(dev)
mm = torch.randint(1<<30, 1<<31, (17,), generator=g, dtype=torch.int64).to(torch.int32).to(dev)
rr = torch.randint(0, 32, (17,), generator=g, dtype=torch.int32).to(dev)
zz = torch.randint(-128, 128, (17,), generator=g, dtype=torch.int32).to(dev)
o2 = kernel.run(a, mm, rr, zz)
r2 = kernel.reference(a, mm, rr, zz)
mism = (o2 != r2)
print("random mismatch count:", mism.sum().item(), "/", mism.numel())
idx = mism.nonzero()[:5]
print("sample mismatch idx:", idx.tolist())
if mism.any():
    i,c = idx[0].tolist()
    print("a=", a[i,c].item(), "m=", mm[c].item(), "r=", rr[c].item(), "z=", zz[c].item(),
          "kernel=", o2[i,c].item(), "ref=", r2[i,c].item())
