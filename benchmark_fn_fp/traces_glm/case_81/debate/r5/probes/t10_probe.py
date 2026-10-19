
import importlib.util, json, torch

spec = importlib.util.spec_from_file_location("k154", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

dev = "cuda"

def fp16_all_finite():
    bits = torch.arange(0, 65536, dtype=torch.int32)
    b = torch.tensor(bits.numpy(), dtype=torch.int16).view(torch.float16).to(torch.float64)
    return b[torch.isfinite(b)]

V = fp16_all_finite()

def ref64(num, den):
    a, b = num.to(torch.float64).unbind(1)
    c, d = den.to(torch.float64).unbind(1)
    s = c*c + d*d
    return torch.stack(((a*c+b*d)/s, (b*c-a*d)/s), dim=1)

def check(num_cpu, den_cpu, tag):
    num = num_cpu.to(dev); den = den_cpu.to(dev)
    out = k.run(num, den).to("cpu").to(torch.float64)
    tgt = ref64(num_cpu, den_cpu)
    err = (out - tgt).abs()
    tol = 1e-7 + 5e-7 * tgt.abs()
    ratio = err / tol
    flat = int(ratio.argmax())
    i = flat // 2; comp = flat % 2
    return {
        "tag": tag, "n": num_cpu.shape[0],
        "n_violations": int((ratio > 1.0).sum()),
        "max_ratio": float(ratio.max()),
        "worst_row_index": i, "worst_component": comp,
        "num_row": num_cpu[i].tolist(), "den_row": den_cpu[i].tolist(),
        "actual": out[i].tolist(), "target": tgt[i].tolist(),
        "err": err[i].tolist(), "tol": tol[i].tolist(),
        "finite_out": bool(torch.isfinite(out).all()),
    }

g = torch.Generator().manual_seed(1)
results = []

# 1) targeted near-cancellation: choose d as nearest fp16 to -a*c/b so true real part tiny
a_t = V[torch.randint(0, len(V), (100000,), generator=g)]
b_t = V[torch.randint(0, len(V), (100000,), generator=g)]
c_t = V[torch.randint(0, len(V), (100000,), generator=g)]
nz = (b_t != 0)
a_t, b_t, c_t = a_t[nz], b_t[nz], c_t[nz]
n = a_t.shape[0]
d_ex = -a_t * c_t / b_t
d_t = torch.tensor(d_ex.numpy(), dtype=torch.float16).to(torch.float64)
bad = ~torch.isfinite(torch.tensor(d_t.tolist(), dtype=torch.float32))
# replace non-finite with extreme
d_t = torch.where(torch.tensor(bad.tolist()), torch.full((n,), 65504.0, dtype=torch.float64), d_t)
num = torch.stack([a_t, b_t], dim=1).to(torch.float16)
den = torch.stack([c_t, d_t], dim=1).to(torch.float16)
mask = ~((den == 0).all(dim=1))
num, den = num[mask], den[mask]
results.append(check(num, den, "targeted_cancellation"))

# 2) uniform random over ALL finite fp16 values
for batch in range(6):
    idx = torch.randint(0, len(V), (200000, 4), generator=g)
    num = V[idx[:, 0:2]].to(torch.float16)
    den = V[idx[:, 2:4]].to(torch.float16)
    mask = ~((den == 0).all(dim=1))
    num, den = num[mask], den[mask]
    results.append(check(num, den, f"uniform_all_fp16_batch{batch}"))

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
