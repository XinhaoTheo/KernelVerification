import torch, sys, json
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
torch.manual_seed(0)

# grid of positive fp16 values
pos = torch.arange(0x0400, 0x7C00, dtype=torch.int32).to(torch.float32).view(torch.int16)  # invalid
# simpler: all positive finite fp16 bit patterns
bits = torch.arange(0, 0x7C01, dtype=torch.int16)
pos_all = torch.cat([bits, torch.tensor([0], dtype=torch.int16)]).to(torch.uint16).to(torch.int16)
pos16 = pos_all.view(torch.float16).to(torch.float32)
pos16 = pos16[torch.isfinite(pos16) & (pos16 >= 0)]
# subnormals included since view handles all patterns
pos16 = torch.unique(pos16)
print("num positive fp16 values:", len(pos16))

worst = -1.0
worst_info = None
count_viol = 0
trials = 0

def check(num, den):
    global worst, worst_info, count_viol, trials
    out = kernel.run(num, den)
    ref = kernel.reference(num.cpu(), den.cpu()).to(dev)
    err = (out - ref).abs()
    tol = 1e-7 + 5e-7*ref.abs()
    viol = err > tol
    cnt = int(viol.sum())
    if cnt:
        count_viol += cnt
        r = ((err - tol)/tol)
        rmasked = torch.where(viol, r, torch.full_like(r, -1.0))
        m = rmasked.max().item()
        if m > worst:
            worst = m
            flat = int(rmasked.argmax())
            i, comp = flat // 2, flat % 2
            worst_info = dict(row=int(i), comp=comp,
                              num=num[i].tolist(), den=den[i].tolist(),
                              out=out[i].tolist(), ref=ref[i].tolist(),
                              err=err[i].tolist(), tol=tol[i].tolist(),
                              excess_ratio=m)
    trials += num.shape[0]

# random combinations over full fp16 grid incl subnormals/extremes, with sign
import random
random.seed(0)
for it in range(400):
    n = 4096
    a = pos16[torch.randint(0, len(pos16), (n,))] * (torch.randint(0,2,(n,))*2-1)
    b = pos16[torch.randint(0, len(pos16), (n,))] * (torch.randint(0,2,(n,))*2-1)
    c = pos16[torch.randint(0, len(pos16), (n,))] * (torch.randint(0,2,(n,))*2-1)
    d = pos16[torch.randint(0, len(pos16), (n,))] * (torch.randint(0,2,(n,))*2-1)
    nz = ~((c == 0) & (d == 0))
    num = torch.stack([a, b], dim=1).to(torch.float16).to(dev)
    den = torch.stack([c, d], dim=1).to(torch.float16).to(dev)
    num = num[nz.to(dev)].contiguous(); den = den[nz.to(dev)].contiguous()
    if num.shape[0] == 0: continue
    check(num, den)

# targeted cancellation: imag numerator b*c - a*d with b*c ~ a*d
# pick b, c, a arbitrary fp16, then d = closest fp16 to a/b*c... hard; instead
# use structured: a = u, d = v, b = u, c = v1, v1 adjacent to v
for it in range(200):
    n = 4096
    idx = torch.randint(0, len(pos16)-1, (n,))
    u = pos16[idx]; v1 = pos16[idx+1]; v2 = pos16[torch.randint(0, len(pos16), (n,))]
    # imag numerator = u*v1 - u*v2? no, structure: a=u, b=u, c=v1, d=v2 -> b*c - a*d = u*(v1 - v2)
    a = u; b = u; c = v1; d = v2
    signs = (torch.randint(0,2,(n,))*2-1).float()
    num = torch.stack([a, b*signs], dim=1).to(torch.float16).to(dev)
    den = torch.stack([c, d], dim=1).to(torch.float16).to(dev)
    check(num, den)

print(json.dumps(dict(trials=trials, violations=count_viol,
                      worst_excess_ratio=worst, worst=worst_info)))
