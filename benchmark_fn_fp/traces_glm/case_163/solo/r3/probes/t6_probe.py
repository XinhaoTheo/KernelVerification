import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def check(vals, label):
    out = k.run(vals.clone()).to(torch.float64)
    ref = k.reference(vals).to(vals.device)
    diff = (out - ref).abs()
    tol = 1e-4 + 1e-6*ref.abs()
    bad = diff > tol
    return {"label": label, "shape": list(vals.shape),
            "max_abs_err": float(diff.max()), "num_violations": int(bad.sum()),
            "finite": bool(torch.isfinite(out).all())}

results = []
for n in (64, 128, 256):
    for seed in range(3):
        v = torch.randn((7, n), generator=torch.Generator().manual_seed(seed)).to(torch.float16).cuda()
        results.append(check(v, f"randn n={n} seed={seed}"))
    # extreme values
    v = torch.full((3, n), 65504.0, dtype=torch.float16).cuda(); results.append(check(v, f"maxval n={n}"))
    v = torch.full((3, n), -65504.0, dtype=torch.float16).cuda(); results.append(check(v, f"-maxval n={n}"))
    v = torch.randn((5, n)).cuda().to(torch.float16) * 1000; results.append(check(v, f"large-mix n={n}"))
    # alternating / single spike / constant
    v = torch.zeros((2, n), dtype=torch.float16).cuda(); v[0,5]=1.0; v[1,:]=1.0
    results.append(check(v, f"spike+const n={n}"))
    v = (torch.arange(n, dtype=torch.float32) % 2 * 2 - 1).to(torch.float16).cuda().unsqueeze(0).repeat(2,1)
    results.append(check(v, f"alternating n={n}"))
# batch extremes
v = torch.randn((256, 128)).to(torch.float16).cuda(); results.append(check(v, "B=256 n=128"))
v = torch.randn((1, 256)).to(torch.float16).cuda(); results.append(check(v, "B=1 n=256"))
# nonzero storage offset
base = torch.randn((256*300+13,), dtype=torch.float16).cuda()
v = base[13:13+7*128].view(7,128).contiguous().to(torch.float16).cuda()
v = base[13:13+7*128].clone().view(7,128)
results.append(check(v, "offset-clone n=128"))
# input preservation
v = torch.randn((4,128)).to(torch.float16).cuda(); vc = v.clone(); _ = k.run(v)
results.append({"label":"input_preserved", "preserved": bool(torch.equal(v, vc))})

viol = [r for r in results if r.get("num_violations",0) or not r.get("finite", True)]
print(json.dumps({"results": results, "violations": viol}, default=str))
