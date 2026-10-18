import importlib.util, json, torch

spec = importlib.util.spec_from_file_location("k154b", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"

def fp16_all_finite():
    bits = torch.arange(0, 65536, dtype=torch.int32)
    b = torch.tensor(bits.numpy(), dtype=torch.int16).view(torch.float16).to(torch.float64)
    return b[torch.isfinite(b)]

V = fp16_all_finite()
ext = [65504.0, -65504.0, 65503.0, 60000.0, -60000.0]
sub = [5.96e-08, 1.19e-07, 2.38e-07, 6e-5]
special_c = ext + sub + [0.0, 1.0]
special_d = ext + sub + [0.0]

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
    return {"tag": tag, "n": int(num_cpu.shape[0]),
        "n_violations": int((ratio > 1.0).sum()),
        "max_ratio": float(ratio.max()),
        "worst_row_index": i, "worst_component": comp,
        "num_row": num_cpu[i].tolist(), "den_row": den_cpu[i].tolist(),
        "actual": out[i].tolist(), "target": tgt[i].tolist(),
        "err": err[i].tolist(), "tol": tol[i].tolist(),
        "finite_out": bool(torch.isfinite(out).all())}

g = torch.Generator().manual_seed(7)
results = []

# 1) each extreme/subnormal denominator pair separately (chunk 1000 -> far below cap)
for c in special_c:
    for d in special_d:
        if c == 0.0 and d == 0.0:
            continue
        idx = torch.randint(0, len(V), (1000, 2), generator=g)
        nn = V[idx].to(torch.float16)
        dd = torch.tensor([[c, d]] * 1000, dtype=torch.float16)
        r = check(nn, dd, f"fixed_den c={c} d={d}")
        results.append(r)

# 2) mixed-magnitude denominators: one component forced to an extreme
for rep in range(4):
    cd = V[torch.randint(0, len(V), (100000, 2), generator=g)].to(torch.float16)
    force = torch.rand(100000, generator=g) < 0.5
    cd[force, 0] = 65504.0
    cd[~force, 1] = -65504.0
    mask = ~((cd == 0).all(dim=1))
    num = V[torch.randint(0, len(V), (100000, 2), generator=g)].to(torch.float16)
    results.append(check(num[mask], cd[mask], f"mixed_extreme_denominator_{rep}"))

mx = max(x["max_ratio"] for x in results)
nv = sum(x["n_violations"] for x in results)
worst = max(results, key=lambda x: x["max_ratio"])
print(json.dumps({"metric": "per-component error/(1e-7+5e-7*|target|)", "total_rows_tested": sum(x["n"] for x in results),
    "overall_max_ratio": mx, "total_violations": nv,
    "worst_batch": worst, "claim": "c2",
    "decisive_signal": "overall_max_ratio > 1.0 confirms; <= 1.0 over extremes/subnormals sample rebuts"}))