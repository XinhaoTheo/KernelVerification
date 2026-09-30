import sys, json, importlib.util
sys.path.insert(0, "/root/pilot_cases/case_87")
spec = importlib.util.spec_from_file_location("kmod", "/root/pilot_cases/case_87/kernel.py")
kmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kmod)

a, b = kmod.make_inputs("cuda")
out = kmod.run(a, b)

# float64 reference on the SAME float32 input values
af = a.to(torch.float64) if False else a.double()
bf = b.double()
ref = torch.empty_like(bf)
h = torch.zeros(bf.shape[1], dtype=torch.float64, device=bf.device)
for t in range(bf.shape[0]):
    h = af[t] * h + bf[t]
    ref[t] = h

out_f = out.double()
diff = out_f - ref
N = out_f.numel()
E = diff.norm() / max(ref.norm(), 0.001 * N ** 0.5)
print(json.dumps({
    "metric": "E (contract L2 relative error)",
    "E": float(E),
    "budget": 0.003,
    "within_budget": bool(E <= 0.003),
    "all_finite": bool(torch.isfinite(out).all().item()),
    "max_abs_err": float(diff.abs().max().item()),
    "ref_norm": float(ref.norm().item()),
    "denom_floor": float(0.001 * (N ** 0.5)),
    "N": int(N),
    "margin_ratio": float((0.003 - E) / 0.003),
    "err_growth_first_last_row_norm": [float(diff[0].norm().item()), float(diff[-1].norm().item())],
}))
