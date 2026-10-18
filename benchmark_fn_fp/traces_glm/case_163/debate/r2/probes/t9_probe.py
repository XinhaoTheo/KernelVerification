import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k163", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(123)
N = 256
g = torch.Generator(device="cpu").manual_seed(5)
rows = [
    torch.full((N,), 65504.0),
    torch.full((N,), -65504.0),
    torch.where(torch.rand(N, generator=g) < 0.5, torch.tensor(65504.0), torch.tensor(-65504.0)),
    (torch.randn(N, generator=g) * 20000).to(torch.float16),
]
vals = torch.stack(rows).to(torch.float16).cuda()
out = k.run(vals)
ref = k.reference(vals)
diff = (out.double() - ref).abs()
tol = 1e-4 + 1e-6*ref.abs()
res = {
    "max_abs_exact": float(ref.abs().max()),
    "max_err": float(diff.max()),
    "max_err_over_tol": float((diff/tol).max()),
    "n_violations": int((diff > tol).sum()),
    "nonfinite_out": int((~torch.isfinite(out)).sum()),
    "worst_location": [int(x) for x in (diff/tol).flatten().argmax() and [(diff/tol).flatten().argmax() // diff.numel(), (diff/tol).flatten().argmax() % diff.numel()]] if diff.numel() else None,
}
print(json.dumps({"metric": "extreme-magnitude tolerance: per-component |actual-exact|/(1e-4+1e-6|exact|) for all-65504, all--65504, mixed-extreme and large-random rows at N=256", "results": res}))
