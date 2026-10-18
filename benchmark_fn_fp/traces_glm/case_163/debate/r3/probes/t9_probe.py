import sys, json, importlib.util
sys.path.insert(0, "/root/cases/case_163")
import kernel, torch

dev = "cuda"
results = {}
for N in (64, 128, 256):
    torch.manual_seed(1234 + N)
    vals = torch.randn(4, N, device=dev).to(torch.float16)
    vals = vals.contiguous()
    out = kernel.run(vals)
    ref = torch.fft.rfft(vals.to(torch.float64), dim=1, norm="backward")
    ref = torch.stack((ref.real, ref.imag), dim=-1)
    actual = out.to(torch.float64)
    tol = 1e-4 + 1e-6 * ref.abs()
    viol = (actual - ref).abs() / tol
    # per-bin max over batch and components
    per_bin = viol.max(dim=0).values.max(dim=-1).values
    results[f"N{N}"] = {
        "max_viol_ratio": float(viol.max()),
        "max_viol_loc": [int(torch.argmax(viol) // (N//2+1) // 2), int(torch.argmax(viol) // 2 % (N//2+1)), int(torch.argmax(viol) % 2)],
        "per_bin_max_ratio": [float(x) for x in per_bin],
        "max_abs_err": float((actual - ref).abs().max()),
        "finite": bool(torch.isfinite(out).all()),
    }
print(json.dumps(results))
