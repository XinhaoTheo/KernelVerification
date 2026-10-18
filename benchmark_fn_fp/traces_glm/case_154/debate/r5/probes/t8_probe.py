
import importlib.util, json, torch

spec = importlib.util.spec_from_file_location("k154", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

dev = "cuda"
all_vals = torch.cat([torch.arange(-2048, 2049)])  # placeholder, replaced below

def fp16_all_finite():
    bits = torch.arange(0, 65536, dtype=torch.int32)
    b = torch.tensor(bits.numpy(), dtype=torch.int16).view(torch.float16).to(torch.float64)
    m = torch.isfinite(b)
    return b[m]

vals = fp16_all_finite()  # all finite fp16 values as fp64, on CPU

def ref64(num, den):
    a, b = num.to(torch.float64).unbind(1)
    c, d = den.to(torch.float64).unbind(1)
    s = c*c + d*d
    return torch.stack(((a*c+b*d)/s, (b*c-a*d)/s), dim=1)

def check(num_cpu, den_cpu, tag):
    n = num_cpu.shape[0]
    num = num_cpu.to(dev); den = den_cpu.to(dev)
    out = k.run(num, den).to("cpu").to(torch.float64)
    tgt = ref64(num_cpu, den_cpu)
    err = (out - tgt).abs()
    tol = 1e-7 + 5e-7 * tgt.abs()
    ratio = (err / tol)
    worst = int(ratio.max())
    i = int(ratio.argmax())
    nviol = int((ratio > 1.0).sum())
    rec = {
        "tag": tag, "n": n, "n_violations": nviol, "max_ratio": float(ratio.max()),
        "worst_row_index": i,
        "num_row": num_cpu[i].tolist(), "den_row": den_cpu[i].tolist(),
        "actual": out[i].tolist(), "target": tgt[i].tolist(),
        "err": err[i].tolist(), "tol": tol[i].tolist(),
        "finite_out": bool(torch.isfinite(out).all()),
    }
    return rec, ratio

g = torch.Generator().manual_seed(1)
results = []

# 1) targeted near-cancellation: a*c ~ -b*d so true real component is tiny
rows = []
V = vals
for _ in range(60000):
    a = V[int(torch.randint(0, len(V), (1,), generator=g))]
    c = V[int(torch.randint(0, len(V), (1,), generator=g))]
    b = V[int(torch.randint(0, len(V), (1,), generator=g))]
    if b == 0 or c == 0:
        rows.append((a, b, c, float(V[int(torch.randint(0, len(V), (1,), generator=g))])))
        continue
    dexact = -a * c / b
    # nearest fp16 value to dexact
    d16 = torch.tensor([dexact], dtype=torch.float16)
    if not torch.isfinite(d16):
        d16 = torch.tensor([65504.0], dtype=torch.float16)
    d = float(d16.to(torch.float64))
    if c == 0.0 and d == 0.0:
        d = 1.0
    rows.append((a, b, c, d))
num = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.float16)
den = torch.tensor([[r[2], r[3]] for r in rows], dtype=torch.float16)
mask = ~((den == 0).all(dim=1))
num, den = num[mask], den[mask]
r, ratio = check(num, den, "targeted_cancellation")
results.append(r)

# 2) uniform random over ALL finite fp16 values (includes extrema/subnormals)
for batch in range(6):
    idx = torch.randint(0, len(V), (200000, 4), generator=g)
    num = V[idx[:, 0:2]].to(torch.float16)
    den = V[idx[:, 2:4]].to(torch.float16)
    mask = ~((den == 0).all(dim=1))
    num, den = num[mask], den[mask]
    r, ratio = check(num, den, f"uniform_all_fp16_batch{batch}")
    results.append(r)

mx = max(x["max_ratio"] for x in results)
nv = sum(x["n_violations"] for x in results)
print(json.dumps({
    "metric": "per-component error / (1e-7 + 5e-7*|target|), max over rows",
    "batches": results,
    "overall_max_ratio": mx,
    "total_violations": nv,
    "claim": "c1 cancellation-heavy inputs exceed tolerance",
    "decisive_signal": "overall_max_ratio > 1.0 confirms; <= 1.0 with large adversarial sample rebuts",
}))
