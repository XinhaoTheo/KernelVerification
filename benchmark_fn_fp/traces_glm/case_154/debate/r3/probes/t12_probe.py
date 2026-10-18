import torch, sys, json
sys.path.insert(0, "/root/cases/case_154")
import kernel as K
import random
random.seed(7)
# Engineered genuine near-cancellation: choose a,b,c,d so that b*c ~ a*d but not equal,
# using non-power-of-two fp16 values so products stress fp32 and the difference is small.
def fp16(x): return torch.tensor([x], dtype=torch.float16).item()
cases = []
vals = [fp16(v) for v in [1.1, 1.7, 2.3, 3.1, 5.7, 7.9, 11.3, 13.7, 100.1, 300.3, 511.7, 1000.5, 2047.7, 4095.3]]
for b in vals:
    for c in vals:
        for a in vals:
            for d in vals:
                bc, ad = b*c, a*d
                if bc == 0 or ad == 0: continue
                if 0 < abs(bc - ad) / max(abs(bc), abs(ad)) < 1e-3:
                    cases.append([a, b, c, d])
                    if len(cases) > 3000: break
            if len(cases) > 3000: break
        if len(cases) > 3000: break
    if len(cases) > 3000: break
if not cases:
    # fallback: random search for near-cancellation quadruples
    for _ in range(200000):
        a, b, c, d = (random.choice(vals) for _ in range(4))
        if a == 0 or b == 0: continue
        # solve d so that b*c ~ a*d: pick d from vals closest to b*c/a
        target = b*c/a
        d = min(vals, key=lambda v: abs(v - target))
        cases.append([a, b, c, d])
        if len(cases) >= 3000: break
num = torch.tensor([[a, b] for a, b, c, d in cases], dtype=torch.float16, device="cuda")
den = torch.tensor([[c, d] for a, b, c, d in cases], dtype=torch.float16, device="cuda")
out = K.run(num, den).double()
ref = K.reference(num.cpu(), den.cpu()).double().cuda()
err = (out - ref).abs()
tol = 1e-7 + 5e-7 * ref.abs()
ok = err <= tol
bad = (~ok).nonzero()
worst = (err / tol).max().item()
print("n_cases:", len(cases), "n_violations:", int((~ok).sum()), "max err/tol ratio:", worst)
if len(bad):
    i = bad[0, 0].item(); comp = bad[0, 1].item()
    print("worst case:", cases[i], "ref:", ref[i, comp].item(), "out:", out[i, comp].item(),
          "err:", err[i, comp].item(), "tol:", tol[i, comp].item())
print(json.dumps({"n_cases": len(cases), "n_violations": int((~ok).sum()), "max_err_over_tol": worst}))