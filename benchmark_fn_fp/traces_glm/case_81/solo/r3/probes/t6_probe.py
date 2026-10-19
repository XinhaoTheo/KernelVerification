import torch, json, sys, os
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
def check(num, den):
    out = kernel.run(num, den)
    ref = kernel.reference(num.cpu(), den.cpu()).to(dev)
    err = (out.double() - ref.double()).abs()
    tol = 1e-7 + 5e-7 * ref.double().abs()
    ratio = (err / tol)
    return out, ref, ratio

results = {}

# Adversarial: near-cancelling products, large magnitudes
worst = {"ratio": 0.0, "case": None}
rows = []
# a*c ~ -b*d with large values: pick c,d large, a = b*t? Use a/b = -d/c ratios exact in fp16
import itertools
found = []
vals = [1024.0, 2048.0, 4096.0, 512.0, 8192.0, 1.0, 2.0, 4.0, 0.5, 3.0, 6.0]
for a in vals:
    for b in vals:
        for c in vals:
            for d in vals:
                if c==0 and d==0: continue
                rn = a*c + b*d
                if rn == 0.0: continue
                if abs(rn) / (abs(a*c)+abs(b*c)+abs(a*d)+abs(b*d)) < 1e-5:
                    found.append((a,b,c,d))
# pick a few hundred worst-conditioned combos
found.sort(key=lambda t: abs(t[0]*t[2]+t[1]*t[3])/(abs(t[0]*t[2])+abs(t[1]*t[3])))
N = min(len(found), 2000)
num = torch.tensor([[f[0],f[1]] for f in found[:N]], dtype=torch.float16, device=dev)
den = torch.tensor([[f[2],f[3]] for f in found[:N]], dtype=torch.float16, device=dev)
out, ref, ratio = check(num, den)
results["adversarial_n"] = N
results["adversarial_max_ratio"] = float(ratio.max())
results["adversarial_violations"] = int((ratio > 1.0).sum())
if float(ratio.max()) > worst["ratio"]:
    i = int(ratio.max().flatten().argmax())
    worst["ratio"] = float(ratio.max())
    r, cc = divmod(i, 2)
    worst["case"] = found[r] + ("real" if cc==0 else "imag")

# Also large random fp16 values
g = torch.Generator(device="cpu").manual_seed(7)
big = torch.empty((20000,2), dtype=torch.float32).uniform_(-65504, 65504, generator=g)
num2 = big.to(torch.float16).to(dev)
den2 = (torch.empty((20000,2), dtype=torch.float32).uniform_(-65504, 65504, generator=g)).to(torch.float16).to(dev)
z = (den2==0).all(dim=1); den2[z,0]=1
out2, ref2, ratio2 = check(num2, den2)
results["random_large_max_ratio"] = float(ratio2.max())
results["random_large_violations"] = int((ratio2>1.0).sum())

# Benign random (make_inputs)
n3, d3 = kernel.make_inputs(dev, size=4093, seed=0)
out3, ref3, ratio3 = check(n3, d3)
results["benign_max_ratio"] = float(ratio3.max())
results["benign_violations"] = int((ratio3>1.0).sum())

results["worst_case"] = worst["case"]
print(json.dumps(results))
