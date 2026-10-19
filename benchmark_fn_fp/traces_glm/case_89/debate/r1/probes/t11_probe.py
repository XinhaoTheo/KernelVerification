import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_162/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
results = {}
# extreme magnitude inputs, several patterns, N=256 and 128
cases = {
 "all65504_N256": torch.full((4, 256), 65504.0, dtype=torch.float16),
 "alt_extreme_N256": torch.tensor([65504.0, -65504.0]*(256*4), dtype=torch.float16).reshape(4, 256),
 "mix_extreme_N256": (torch.randn(4, 256, generator=torch.Generator(device="cpu").manual_seed(5)).to(torch.float16) * 65504.0).clamp(-65504, 65504).to(torch.float16),
 "alt_extreme_N128": torch.tensor([65504.0, -65504.0]*(128*4), dtype=torch.float16).reshape(4, 128),
}
for name, vals_cpu in cases.items():
    vals = vals_cpu.cuda()
    out = k.run(vals)
    ref = torch.fft.rfft(vals.double().cpu(), dim=1)
    refc = torch.stack((ref.real, ref.imag), dim=-1).cuda()
    tol = 1e-4 + 1e-6*refc.abs()
    err = (out.double() - refc).abs()
    rel = (err / refc.abs().clamp_min(1e-30)).max().item()
    results[name] = {
        "max_abs_err": err.max().item(),
        "max_rel_err": rel,
        "tol_violations": int((err > tol).sum()),
        "nonfinite_out": bool((~torch.isfinite(out)).any()),
        "max_out_mag": out.abs().max().item(),
    }
print(json.dumps(results))