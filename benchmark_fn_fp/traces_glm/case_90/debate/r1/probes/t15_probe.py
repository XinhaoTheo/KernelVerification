import torch, sys, os, json
sys.path.insert(0, os.environ.get("CASE_DIR", "/root/cases/case_163"))
import kernel
dev = "cuda"
worst = 0.0; worst_case = None; fails = []
for N in (64, 128, 256):
    cases = {}
    cases["const65504"] = torch.full((4, N), 65504.0)
    cases["const_neg65504"] = torch.full((2, N), -65504.0)
    cases["alt65504"] = torch.stack([torch.tensor([65504.0 if (i%2==0) else -65504.0 for i in range(N)], dtype=torch.float64) for _ in range(4)])
    g = torch.Generator().manual_seed(1)
    mix = torch.randn(8, N, generator=g).abs() * 0.5  # safe scale
    mix[0, 3] = 65504.0; mix[1, 5] = -65504.0
    mix[2, :] = 5.96e-8  # smallest fp16 subnormal
    mix[3, :] = torch.where(torch.arange(N) % 2 == 0, 65504.0, -65504.0)
    mix[4, :] = 65504.0; mix[4, ::2] = -65504.0  # near-alternating extremes
    cases["mix"] = mix
    cases["subnorm"] = torch.full((2, N), 5.96e-8)
    cases["zerorow"] = torch.zeros((2, N))
    for name, m in cases.items():
        m16 = m.to(torch.float16).to(dev)
        assert torch.isfinite(m16).all(), (name, N)
        out = kernel.run(m16)
        ref = torch.fft.rfft(m16.to(torch.float64), dim=1, norm="backward")
        ref = torch.stack((ref.real, ref.imag), dim=-1)
        err = (out.double() - ref).abs()
        tol = 1e-4 + 1e-6 * ref.abs()
        v = int((err > tol).sum().item())
        r = (err / tol).max().item()
        if v > 0:
            fails.append({"case": name, "N": N, "count": v, "max_ratio": r,
                          "max_abs_err": err.max().item()})
        if r > worst:
            worst = r; worst_case = f"{name}_N{N}"
        nonfinite = int((~torch.isfinite(out)).sum().item())
        if nonfinite:
            fails.append({"case": name, "N": N, "nonfinite": nonfinite})
print(json.dumps({"claim": "c2", "fails": fails, "worst_ratio": worst, "worst_case": worst_case}))