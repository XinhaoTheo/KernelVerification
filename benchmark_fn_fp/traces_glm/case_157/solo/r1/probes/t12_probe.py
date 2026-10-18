import torch, math, sys
sys.path.insert(0, "/root/cases/case_157")
import kernel

TARGET = 1 << 61  # halfway for r=31: bias 2^(30+31)=2^61
found = []
for m in range((1<<30)+1, (1<<31), 2):
    rem = TARGET % m
    if 0 < rem < 256:  # P below halfway by <256 (half of product-ulp 512 near 2^61) -> RN rounds product up to 2^61
        a = (TARGET - rem) // m
        if 0 < a <= 0x7fffffff:
            found.append((a, m, 31))
    if len(found) >= 16:
        break
print("constructible cases:", len(found), found[:4])

def exact(a, m, r, z=0):
    P = a * m
    R = (P + (1 << (30 + r))) >> (31 + r)
    return max(-128, min(127, R + z))

n = len(found)
if n:
    acc = torch.tensor([f[0] for f in found], dtype=torch.int32, device="cuda").view(n,1)
    mult = torch.tensor([f[1] for f in found], dtype=torch.int32, device="cuda")
    sh = torch.full((n,), 31, dtype=torch.int32, device="cuda")
    zp = torch.zeros(n, dtype=torch.int32, device="cuda")
    out = kernel.run(acc, mult, sh, zp).cpu()
    ref = kernel.reference(acc, mult, sh, zp).cpu()
    print("kernel:", out.flatten().tolist())
    print("ref   :", ref.flatten().tolist())
    mism = (out != ref)
    print("GPU mismatch:", bool(mism.any()), "count:", int(mism.sum()), "/", n)
    for i,(a,m,r) in enumerate(found):
        print(i, "a=",a, "m=",m, "P=",a*m, "P_dist_below_2^61:", TARGET - a*m,
              "exact=", exact(a,m,31), "gpu_kernel=", out[i,0].item(), "gpu_ref=", ref[i,0].item())
