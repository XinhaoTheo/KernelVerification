import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_162/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
results = {}
for N in (64, 128, 256):
    vals = torch.randn(5, N, generator=torch.Generator(device="cpu").manual_seed(11)).to(torch.float16).cuda()
    out = k.run(vals)
    # exact Nyquist: sum_j (-1)^j x[j] computed in float64 on the exact fp16 values
    signs = torch.tensor([(-1.0)**j for j in range(N)], dtype=torch.float64)
    nyq = (vals.double().cpu() * signs).sum(dim=1)
    got = out[:, N//2, 0].double().cpu()
    imag_got = out[:, N//2, 1].double().cpu()
    tol = 1e-4 + 1e-6*nyq.abs()
    err = (got - nyq).abs()
    results[f"N{N}"] = {
        "nyquist_real_max_err": err.max().item(),
        "tolerance": tol.min().item(),
        "violations": int((err > tol).sum()),
        "imag_max_abs": imag_got.abs().max().item(),
        "ref_nyquist": nyq.tolist()[:3],
        "got_nyquist": got.tolist()[:3],
    }
print(json.dumps(results))