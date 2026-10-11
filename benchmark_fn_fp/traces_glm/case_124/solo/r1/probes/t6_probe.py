import torch, sys
sys.path.insert(0, "/root/cases/case_124")
import kernel as K

dev = "cuda"
fails = []
maxerr = 0.0

def check(acc, mult, shifts, zeros, scales, tag):
    global maxerr
    out = K.run(acc, mult, shifts, zeros, scales)
    ref = K.reference(acc, mult, shifts, zeros, scales)
    diff = (out - ref).abs().max().item()
    maxerr = max(maxerr, diff)
    if diff > 1e-6:
        bad = ((out - ref).abs() > 1e-6).nonzero()
        i, c = bad[0].tolist()
        fails.append((tag, diff, int(i), int(c), int(acc[i,c]), int(mult[c]), int(shifts[c]), int(zeros[c]), float(scales[c]), out[i,c].item(), ref[i,c].item()))

# random sweeps: many seeds, varying r coverage
for seed in range(8):
    acc, mult, shifts, zeros, scales = K.make_inputs(device=dev, rows=29, channels=64, seed=seed)
    check(acc, mult, shifts, zeros, scales, f"rand{seed}")

# r exhaustive per channel
acc, mult, shifts, zeros, scales = K.make_inputs(device=dev, rows=64, channels=32, seed=99)
shifts = torch.arange(32, dtype=torch.int32, device=dev)
zeros = torch.zeros(32, dtype=torch.int32, device=dev)
scales = torch.ones(32, dtype=torch.float32, device=dev)
check(acc, mult, shifts, zeros, scales, "r-exhaust")
# with z=127 and z=-128
check(acc, mult, shifts, torch.full((32,),127,dtype=torch.int32,device=dev), scales, "r-z127")
check(acc, mult, shifts, torch.full((32,),-128,dtype=torch.int32,device=dev), scales, "r-zm128")
# z=-128 with acc large positive (R big) forces clamp interplay
acc2 = torch.full((64,32), 2**31-1, dtype=torch.int32, device=dev)
check(acc2, mult, shifts, torch.full((32,),-128,dtype=torch.int32,device=dev), scales, "clamp-pos")
acc3 = torch.full((64,32), -2**31, dtype=torch.int32, device=dev)
check(acc3, mult, shifts, torch.full((32,),127,dtype=torch.int32,device=dev), scales, "clamp-neg")

# extreme acc values
for a in [2**31-1, -2**31, 0, 1, -1, 0x40000000, -0x40000000, 0x3fffffff, -0x3fffffff, 0x80000000 if False else -(2**30), 12345, -12345]:
    accv = torch.full((32,32), a, dtype=torch.int32, device=dev)
    m = torch.full((32,), 2**31-1, dtype=torch.int32, device=dev)
    for r in range(32):
        sh = torch.full((32,), r, dtype=torch.int32, device=dev)
        check(accv, m, sh, zeros[:32], scales[:32], f"ext-a{a}-r{r}")

# half-tie constructions: need P + 2^30 == k*2^31 + 2^30 exactly, i.e. P = k*2^31
# choose m = 2^31-1, find a such that a*m % 2^31 == 2^31-... simpler: m=2^30, a odd => P = a*2^30, half ties
m = torch.full((32,), 2**30, dtype=torch.int32, device=dev)
for a in [1,3,5,-1,-3,-5, 2**31-1, -2**31, 1234567, -1234567]:
    accv = torch.full((32,32), a, dtype=torch.int32, device=dev)
    for r in range(32):
        sh = torch.full((32,), r, dtype=torch.int32, device=dev)
        check(accv, m, sh, zeros[:32], scales[:32], f"tie-a{a}-r{r}")

# random m diversity with r=1 specifically
g = torch.Generator().manual_seed(7)
for trial in range(5):
    accv = torch.randint(-(2**31), 2**31, (128,64), dtype=torch.int64, generator=g).to(torch.int32).to(dev)
    mv = torch.randint(2**30, 2**31, (64,), dtype=torch.int64, generator=g).to(torch.int32).to(dev)
    for r in [0,1,2,3,31]:
        sh = torch.full((64,), r, dtype=torch.int32, device=dev)
        zv = torch.randint(-128,128,(64,), dtype=torch.int32, generator=g).to(dev)
        sc = torch.pow(torch.tensor(2.0), torch.randint(-8,1,(64,), generator=g).to(dev)).float()
        check(accv, mv, sh, zv, sc, f"mix{trial}-r{r}")

print(f"maxerr={maxerr!r} num_failures={len(fails)}")
for f in fails[:10]:
    print(f)
import json
print(json.dumps({"maxerr": maxerr, "failures": len(fails), "metric": "max abs error vs kernel.reference (exact Python-int two-stage definition)"}))
