import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_162/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
results = {}
def make(pattern, B, N):
    if pattern == "all":
        return torch.full((B, N), 65504.0, dtype=torch.float16)
    if pattern == "alt":
        return torch.tensor([65504.0, -65504.0]*(N//2)*B, dtype=torch.float16).reshape(B, N)
    if pattern == "randext":
        g = torch.Generator(device="cpu").manual_seed(5)
        return (torch.randn(B, N, generator=g).sign() * 65504.0).to(torch.float16)
    if pattern == "mix":
        g = torch.Generator(device="cpu").manual_seed(7)
        return (torch.randn(B, N, generator=g).to(torch.float16) * 65504.0).clamp(-65504, 65504).to(torch.float16)
for N in (64, 128, 256):
    for pattern in ("all", "alt", "randext", "mix"):
        vals = make(pattern, 4, N).cuda()
        out = k.run(vals)
        ref = torch.fft.rfft(vals.double().cpu(), dim=1)
        refc = torch.stack((ref.real, ref.imag), dim=-1).cuda()
        tol = 1e-4 + 1e-6*refc.abs()
        err = (out.double() - refc).abs()
        results[f"{pattern}_N{N}"] = {
            "max_abs_err": err.max().item(),
            "tol_violations": int((err > tol).sum()),
            "nonfinite_out": bool((~torch.isfinite(out)).any()),
            "max_out_mag": out.abs().max().item(),
        }
print(json.dumps(results))