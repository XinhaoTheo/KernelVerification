import sys, json, torch
sys.path.insert(0, "/root/cases/case_156")
import kernel as K
dev = "cuda"
# Negative AND positive exact-halfway products with legal m in [2^30, 2^31-1].
# With m = 2^30 and a = -(2j+1)*2^r, a*m = -(2j+1)*2^(30+r), quotient = -(2j+1)/2 (exact negative half-integer).
cases = []
ms = [1 << 30, (1 << 30) + 1, (1 << 31) - 1]
for r in range(0, 32):
    for j in range(0, 60):
        a = -(2 * j + 1) * (1 << r)
        if not (-(1 << 31) <= a <= (1 << 31) - 1):
            continue
        for m in ms:
            # product = a*m; halfway iff a*m = odd * 2^(30+r)
            P = a * m
            if P % (1 << (30 + r)) == 0 and (P // (1 << (30 + r))) % 2 != 0:
                cases.append((a, m, r))
        # positive halfway
        a2 = (2 * j + 1) * (1 << r)
        if -(1 << 31) <= a2 <= (1 << 31) - 1:
            for m in ms:
                P = a2 * m
                if P % (1 << (30 + r)) == 0 and (P // (1 << (30 + r))) % 2 != 0:
                    cases.append((a2, m, r))
cases = list(dict.fromkeys(cases))
C = len(cases)
acc = torch.tensor([c[0] for c in cases], dtype=torch.int64).to(torch.int32).to(dev).reshape(1, C)
mul = torch.tensor([c[1] for c in cases], dtype=torch.int64).to(torch.int32).to(dev)
rsh = torch.tensor([c[2] for c in cases], dtype=torch.int32).to(dev)
zp = torch.zeros(C, dtype=torch.int32, device=dev)
out = K.run(acc, mul, rsh, zp)
ref = K.reference(acc, mul, rsh, zp)
mism = (out != ref).sum().item()
detail = []
for i, (a, m, r) in enumerate(cases):
    q = a * m / (1 << (31 + r))
    R = (a * m + (1 << (30 + r))) >> (31 + r)
    if out[0, i].item() != ref[0, i].item():
        detail.append({"a": a, "m": m, "r": r, "quotient": q, "kernel": out[0, i].item(), "ref": ref[0, i].item(), "math_floor_R": R})
print(json.dumps({"metric": "exact_int8_equality_on_exact_halfway_cases",
  "n_cases": C, "mismatches": mism, "mismatch_detail": detail[:20],
  "sample": [{"a": c[0], "m": c[1], "r": c[2], "quotient": c[0]*c[1]/(1<<(31+c[2])),
              "R": (c[0]*c[1]+(1<<(30+c[2])))>>(31+c[2]), "kernel": out[0,i].item()}
             for i, c in enumerate(cases[:8])]}))