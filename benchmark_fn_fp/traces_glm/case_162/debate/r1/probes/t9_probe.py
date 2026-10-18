import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_162/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
results = {}
for N in (64, 128, 256):
    worst = 0.0; worst_loc = None; any_fail = False; nonfinite = False
    for seed in range(3):
        vals = torch.randn(7, N, generator=torch.Generator(device="cpu").manual_seed(seed)).to(torch.float16).cuda()
        out = k.run(vals)
        ref = torch.fft.rfft(vals.double().cpu(), dim=1).cuda()
        refc = torch.stack((ref.real, ref.imag), dim=-1)
        tol = 1e-4 + 1e-6*refc.abs()
        err = (out.double() - refc).abs()
        nonfinite |= bool((~torch.isfinite(out)).any())
        fail = err > tol
        any_fail |= bool(fail.any())
        m = err.max().item()
        if m > worst:
            worst = m; idx = (err == err.max()).nonzero()[0].tolist(); worst_loc = idx
    results[f"N{N}"] = {"worst_abs_err": worst, "worst_loc": worst_loc, "tol_violation": any_fail, "nonfinite_output": nonfinite}
print(json.dumps(results))