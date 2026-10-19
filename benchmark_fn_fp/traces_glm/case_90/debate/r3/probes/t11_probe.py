import sys, json
sys.path.insert(0, "/root/cases/case_163")
import kernel, torch

dev = "cuda"
results = {}
for N in (64, 128, 256):
    M = 65504
    rows = [
        torch.full((N,), M, device=dev),          # all +65504
        torch.full((N,), M, device=dev) * (torch.arange(N, device=dev) % 2 * -2 + 1),  # alternating
        torch.zeros(N, device=dev),               # all zero
    ]
    # row with exact X[k]=0 for chosen k: e.g. sin wave with integer period pairs summing to zero
    # use x[j] = cos(2*pi*f*j/N) type rows: X[0]=0 for balanced +1/-1 pattern
    rows.append((torch.arange(N, device=dev) % 2 * 2 - 1).to(torch.float16))  # alternating +-1: X[0]=0
    # signed-zero subnormal mix
    r = torch.arange(N, device=dev).to(torch.float16) * 6e-8
    r[::2] = -0.0
    rows.append(r)
    vals = torch.stack(rows).to(torch.float16).contiguous()
    assert torch.isfinite(vals).all()
    out = kernel.run(vals)
    ref = torch.fft.rfft(vals.to(torch.float64), dim=1, norm="backward")
    ref = torch.stack((ref.real, ref.imag), dim=-1)
    act = out.to(torch.float64)
    tol = 1e-4 + 1e-6 * ref.abs()
    viol = (act - ref).abs() / tol
    per_row = viol.max(dim=1).values.max(dim=-1).values
    results[f"N{N}"] = {
        "max_viol_ratio": float(viol.max()),
        "per_row_max_ratio": [float(x) for x in per_row],
        "max_abs_err": float((act - ref).abs().max()),
        "nonfinite": bool((~torch.isfinite(out)).any()),
    }
print(json.dumps(results))
