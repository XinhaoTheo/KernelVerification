import sys, json, torch
sys.path.insert(0, "/root/cases/case_156")
import kernel as K
dev = "cuda"
# Exact-halfway products: a*m = odd * 2^(30+r), i.e. quotient a*m/2^(31+r) is an odd/2 half-integer.
cases = []
ms = [1 << 30, (1 << 30) + 1, (1 << 31) - 1]
for r in range(0, 32):
    for j in range(0, 8):
        for sgn in (-1, 1):
            a = sgn * (2 * j + 1) * (1 << r)
            if not (-(1 << 31) <= a <= (1 << 31) - 1):
                continue
            for m in ms:
                P = a * m
                if P % (1 << (30 + r)) == 0 and (P // (1 << (30 + r))) % 2 != 0:
                    cases.append((a, m, r))
cases = list(dict.fromkeys(cases))
# batch into chunks of at most 256 channels (C<=257 legal)
mism = 0
detail = []
samples = []
CH = 256
for s in range(0, len(cases), CH):
    chunk = cases[s:s+CH]
    C = len(chunk)
    acc = torch.tensor([c[0] for c in chunk], dtype=torch.int64).to(torch.int32).to(dev).reshape(1, C)
    mul = torch.tensor([c[1] for c in chunk], dtype=torch.int64).to(torch.int32).to(dev)
    rsh = torch.tensor([c[2] for c in chunk], dtype=torch.int32).to(dev)
    zp = torch.zeros(C, dtype=torch.int32, device=dev)
    out = K.run(acc, mul, rsh, zp)
    ref = K.reference(acc, mul, rsh, zp)
    d = (out != ref).flatten()
    mism += int(d.sum().item())
    for i in range(C):
        a, m, r = chunk[i]
        if d[i].item():
            detail.append({"a": a, "m": m, "r": r, "quotient": a*m/(1<<(31+r)),
                           "kernel": out[0, i].item(), "ref": ref[0, i].item(),
                           "math_floor_R": (a*m + (1 << (30+r))) >> (31+r)})
        if len(samples) < 8:
            samples.append({"a": a, "m": m, "r": r, "quotient": a*m/(1<<(31+r)),
                            "math_floor_R": (a*m + (1 << (30+r))) >> (31+r),
                            "kernel": out[0, i].item(), "ref": ref[0, i].item()})
print(json.dumps({"metric": "exact_int8_equality_on_exact_halfway_cases",
  "n_cases": len(cases), "mismatches": mism, "mismatch_detail": detail[:20], "samples": samples}))