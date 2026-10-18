import torch, sys, json, itertools, random
sys.path.insert(0, "/root/cases/case_154")
import kernel as K

E = 65504.0
# large-magnitude combinations, including extrema, plus a random large-magnitude sweep
cases = []
for a in [E, 32768.0, 4096.0]:
    for c in [E, 32768.0, 4096.0, 2048.0]:
        for b in [0.0, 1.0, E]:
            for d in [0.0, 1.0, E]:
                if c == 0 and d == 0: continue
                cases.append([a, b, c, d])
random.seed(1)
for _ in range(2000):
    a = random.choice([E, -E, 2**random.randint(11, 15), random.random()*1000])
    b = random.choice([0.0, 2**random.randint(11, 15), random.random()*1000, -2**14])
    c = random.choice([E, -E, 2**random.randint(11, 15), random.random()*1000])
    d = random.choice([0.0, 2**random.randint(11, 15), random.random()*1000, -2**14])
    if c == 0 and d == 0: c = 1.0
    cases.append([a, b, c, d])

num = torch.tensor([[a, b] for a, b, c, d in cases], dtype=torch.float16, device="cuda")
den = torch.tensor([[c, d] for a, b, c, d in cases], dtype=torch.float16, device="cuda")
out = K.run(num, den).double()
ref = K.reference(num.cpu(), den.cpu()).double().cuda()
err = (out - ref).abs()
tol = 1e-7 + 5e-7 * ref.abs()
ok = err <= tol
bad = (~ok).nonzero()
print("n_violations:", int((~ok).sum()))
if len(bad):
    i = bad[0, 0].item(); comp = bad[0, 1].item()
    print("worst case:", cases[i], "ref:", ref[i, comp].item(), "out:", out[i, comp].item(),
          "err:", err[i, comp].item(), "tol:", tol[i, comp].item())
print(json.dumps({"n_violations": int((~ok).sum()), "n_cases": len(cases)}))