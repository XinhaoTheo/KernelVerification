import sys, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_125/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

dev = "cuda"
# Channels: one per shift value 0..31, plus a couple of extras
shifts = torch.arange(0, 32, dtype=torch.int32)
C = 32
mult = torch.full((C,), 2**31 - 1, dtype=torch.int32)
# vary multipliers across channels
g = torch.Generator("cpu").manual_seed(1)
mult = torch.randint(2**30, 2**31, (C,), dtype=torch.int64, generator=g).to(torch.int32)
zeros = torch.tensor([0,127,-128] * 11, dtype=torch.int32)[:C]
exps = torch.tensor([-8,-7,-6,-5,-4,-3,-2,-1,0] * 4, dtype=torch.int32)[:C]
scales = torch.pow(torch.tensor(2.0), exps.to(torch.float64)).to(torch.float32)

def H_exact(a, m):
    P = int(a) * int(m)
    return (P + (1 << 30)) // (1 << 31)

# Build adversarial accumulator values per channel
rows = []
# 1) extremes
rows.append([(-2**31)] * C); rows.append([2**31 - 1] * C); rows.append([0]*C)
# 2) exact R-stage tie accumulators (positive and negative) for each channel
rng = torch.Generator("cpu").manual_seed(2)
for trial in range(3):
    tie_pos, tie_neg = [], []
    for c in range(C):
        r = int(shifts[c]); m = int(mult[c])
        found_p, found_n = None, None
        for _ in range(20000):
            a = int(torch.randint(-(2**31), 2**31, (1,), dtype=torch.int64, generator=g).item())
            h = H_exact(a, m)
            if r > 0:
                if found_p is None and h > 0 and h % (1 << r) == (1 << (r - 1)):
                    found_p = a
                if found_n is None and h < 0 and (-h) % (1 << r) == (1 << (r - 1)):
                    found_n = a
            else:
                if found_p is None: found_p = a
                if found_n is None: found_n = -a
            if found_p is not None and found_n is not None: break
        tie_pos.append(found_p if found_p is not None else 0)
        tie_neg.append(found_n if found_n is not None else 0)
    rows.append(tie_pos); rows.append(tie_neg)
# 3) values near saturation boundaries after rounding
rows.append([])
for c in range(C):
    m = int(mult[c])
    # find a such that H (or R) lands near +-128-z
    vals = []
    for target in (127, 128, 129, -127, -128, -129, 255, 256, -255, -256):
        found = None
        for _ in range(30000):
            a = int(torch.randint(-(2**31), 2**31, (1,), dtype=torch.int64, generator=g).item())
            h = H_exact(a, m)
            if h == target: found = a; break
        vals.append(found if found is not None else 0)
    rows[-1] = vals
acc = torch.tensor(rows, dtype=torch.int32)
acc = torch.cat([acc, torch.randint(-(2**31), 2**31, (200, C), dtype=torch.int64, generator=torch.Generator("cpu").manual_seed(3)).to(torch.int32)], 0).to(dev)

params = tuple(t.to(dev) for t in (mult, shifts, zeros, scales))
out = k.run(acc, *params)
ref = k.reference(acc, *params)
diff = (out.double() - ref.double()).abs()
mism = (diff > 1e-6).sum().item()
print("shape", tuple(acc.shape), "max_abs_err", diff.max().item(), "mismatches", mism)
if mism:
    idx = (diff > 1e-6).nonzero()[:10]
    for i, j in idx.tolist():
        print("MISMATCH", i, j, "a=", int(acc[i,j]), "m=", int(mult[j]), "r=", int(shifts[j]), "z=", int(zeros[j]), "s=", float(scales[j]),
              "got=", out[i,j].item(), "ref=", ref[i,j].item(), "H=", H_exact(int(acc[i,j]), int(mult[j])))
import json
print(json.dumps({"max_abs_err": diff.max().item(), "mismatches": mism, "shape": list(acc.shape)}))
