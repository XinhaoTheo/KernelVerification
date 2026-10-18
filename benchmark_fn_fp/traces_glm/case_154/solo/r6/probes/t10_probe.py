import torch, json, sys, random
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(1234)
viol = 0; total = 0; worst = 0.0; worst_info = None; nonfinite = 0

def trial(num, den, tag):
    global viol, total, worst, worst_info, nonfinite
    out = kernel.run(num, den)
    ref = kernel.reference(num, den).to(torch.float64)
    err = (out.to(torch.float64) - ref).abs()
    tol = 1e-7 + 5e-7 * ref.abs()
    nonfinite += int((~torch.isfinite(out)).sum().item())
    ratio = (err / tol)
    m = float(ratio.max().item())
    total += err.numel()
    v = int((err > tol).sum().item())
    viol += v
    if m > worst:
        worst = m
        i = int(ratio.argmax().item() // 2)
        worst_info = (tag, num[i].tolist(), den[i].tolist(), out[i].tolist(), ref[i].tolist())

# 1) uniform random finite fp16 bit patterns (subnormals, zeros, extremes)
for i in range(20):
    n = (torch.randint(0, 65536, (8192, 2), dtype=torch.int32, generator=g) - 32768).to(torch.int16)
    num = n.view(torch.float16).to(dev)
    d = (torch.randint(0, 65536, (8192, 2), dtype=torch.int32, generator=g) - 32768).to(torch.int16)
    den = d.view(torch.float16).to(dev)
    # drop non-finite rows and all-zero denominator rows
    ok = torch.isfinite(num).all(1) & torch.isfinite(den).all(1) & ((den != 0).any(1))
    num, den = num[ok], den[ok]
    if num.shape[0]: trial(num, den, "randbits")

# 2) cancellation-prone pairs across all exponents
vals = []
e = -14
while e <= 15:
    vals.append(torch.tensor(2.0**e, dtype=torch.float16).item())
    vals.append(torch.tensor(2.0**e * 1.41421356, dtype=torch.float16).item())
    vals.append(torch.tensor(2.0**e * 1.00048828125, dtype=torch.float16).item())
    e += 1
V = [v for v in vals if torch.isfinite(torch.tensor(v, dtype=torch.float16)).item()]
random.seed(7)
rows_n, rows_d = [], []
for it in range(400):
    a = V[random.randrange(len(V))]; c = V[random.randrange(len(V))]
    b = V[random.randrange(len(V))]; d = V[random.randrange(len(V))]
    if random.random() < 0.5: b = -b
    if random.random() < 0.5: d = -d
    rows_n.append((a, b)); rows_d.append((c, d))
trial(torch.tensor(rows_n, dtype=torch.float16, device=dev),
      torch.tensor(rows_d, dtype=torch.float16, device=dev), "cancel")

# 3) extremes grid
sub = 5.960464477539063e-08; big = 65504.0
combos = [(a, b, c, dd) for a in [big, -big, 1.0, sub, 0.0] for b in [big, -big, 1.0, sub, 0.0]
          for c in [sub, -sub, big, 1.0] for dd in [sub, -sub, big, 1.0] if not (c == 0 and dd == 0)]
trial(torch.tensor([[r[0], r[1]] for r in combos], dtype=torch.float16, device=dev),
      torch.tensor([[r[2], r[3]] for r in combos], dtype=torch.float16, device=dev), "extremes")

# 4) make_inputs example
n, d = kernel.make_inputs(device=dev, size=4093, seed=0)
trial(n, d, "make_inputs")

print(json.dumps({
    "metric": "componentwise |actual-target| / (1e-7+5e-7|target|) over full finite fp16 domain sampling",
    "components_checked": total,
    "violating_components": viol,
    "nonfinite_outputs": nonfinite,
    "max_violation_ratio": worst,
    "worst_case_info": worst_info,
    "any_violation": viol > 0,
}))
