import torch, math, sys
sys.path.insert(0, "/root/cases/case_157")
import kernel

def kern_emul(a, m, r, z=0):
    pf = float(a) * float(m)
    scaled = pf / float(1 << (31 + r))
    rounded = math.floor(scaled + 0.5)
    bounded = max(-255, min(255, rounded))
    return max(-128, min(127, bounded + z))

def exact(a, m, r, z=0):
    P = a * m
    R = (P + (1 << (30 + r))) >> (31 + r)
    return max(-128, min(127, R + z))

rng = torch.Generator().manual_seed(7)
found = []
for trial in range(300000):
    a = int(torch.randint(-(1<<31), 1<<31, (1,), generator=rng).item())
    m = int(torch.randint(1<<30, 1<<31, (1,), generator=rng).item())
    for r in (29, 30, 31):
        k, e = kern_emul(a, m, r), exact(a, m, r)
        if k != e:
            found.append((a, m, r, k, e))
            break
    if len(found) >= 8:
        break
print("emulation mismatches found:", len(found))
for f in found[:8]: print(f)

if found:
    vals = [f[0] for f in found]
    rs = [f[2] for f in found]
    ms = [f[1] for f in found]
    n = len(vals)
    acc = torch.tensor(vals, dtype=torch.int32, device="cuda").view(n, 1)
    mult = torch.tensor(ms, dtype=torch.int32, device="cuda")
    sh = torch.tensor(rs, dtype=torch.int32, device="cuda")
    zp = torch.zeros(n, dtype=torch.int32, device="cuda")
    out = kernel.run(acc, mult, sh, zp).cpu()
    ref = kernel.reference(acc, mult, sh, zp).cpu()
    print("kernel:", out.flatten().tolist())
    print("ref   :", ref.flatten().tolist())
    print("GPU mismatch:", not torch.equal(out, ref))
    for i,(a,m,r,k,e) in enumerate(found):
        print(i, "a=",a,"m=",m,"r=",r,"emul_k=",k,"exact=",e,
              "gpu_kernel=",out[i,0].item(),"gpu_ref=",ref[i,0].item())
else:
    g = torch.Generator().manual_seed(11)
    a = torch.randint(-(1<<31), 1<<31, (4096, 257), generator=g, dtype=torch.int64).to(torch.int32).cuda()
    mm = torch.randint(1<<30, 1<<31, (257,), generator=g, dtype=torch.int64).to(torch.int32).cuda()
    rr = torch.randint(24, 32, (257,), generator=g, dtype=torch.int32).cuda()
    zz = torch.randint(-128, 128, (257,), generator=g, dtype=torch.int32).cuda()
    o = kernel.run(a, mm, rr, zz); rf = kernel.reference(a, mm, rr, zz)
    mism = (o != rf)
    print("random large-r mismatch:", mism.sum().item(), "/", mism.numel())
    if mism.any():
        idx = mism.nonzero()[:3]
        for i,c in idx.tolist():
            print("a=",a[i,c].item(),"m=",mm[c].item(),"r=",rr[c].item(),"z=",zz[c].item(),
                  "kernel=",o[i,c].item(),"ref=",rf[i,c].item())
