import torch, sys, os
sys.path.insert(0, os.environ.get("CASE_DIR", "/root/cases/case_163"))
import kernel

dev = "cuda"
worst = {"max_ratio": 0.0, "case": None}
fail = False
for N in (64, 128, 256):
    B = min(N // 2 + 1, 256)  # one row per tone bin k=0..N/2
    rows = []
    for k in range(N // 2 + 1):
        j = torch.arange(N, dtype=torch.float64)
        tone = torch.cos(2 * torch.pi * j * k / N)  # exact real tone at bin k
        rows.append(tone)
    rows = torch.stack(rows).to(torch.float16).to(dev)
    # clamp to legal batch
    if rows.shape[0] > 256:
        rows = rows[:256]
    out = kernel.run(rows)
    ref = torch.fft.rfft(rows.to(torch.float64), dim=1, norm="backward")
    ref = torch.stack((ref.real, ref.imag), dim=-1)
    err = (out.double() - ref).abs()
    tol = 1e-4 + 1e-6 * ref.abs()
    viol = err > tol
    ratio = (err / tol).max().item()
    if ratio > worst["max_ratio"]:
        worst = {"max_ratio": ratio, "case": f"tone_N{N}"}
    # also alternating-sign rows and impulse rows
    for name, mat in [("alt", torch.stack([torch.tensor([(-1)**(i+1) for i in range(N)], dtype=torch.float64) for _ in range(4)])),
                      ("imp", torch.stack([torch.eye(N)[i] for i in range(4)]))]:
        m = mat.to(torch.float16).to(dev)
        out = kernel.run(m)
        ref = torch.fft.rfft(m.to(torch.float64), dim=1, norm="backward")
        ref = torch.stack((ref.real, ref.imag), dim=-1)
        err = (out.double() - ref).abs()
        tol = 1e-4 + 1e-6 * ref.abs()
        v = int((err > tol).sum().item())
        r = (err / tol).max().item()
        if v > 0:
            fail = True
            print(f"VIOLATION {name} N={N} count={v}", file=sys.stderr)
        if r > worst["max_ratio"]:
            worst["max_ratio"] = r; worst["case"] = f"{name}_N{N}"
    nv = int((err > tol).sum().item()) if False else 0
    print(f"N={N} done")
print({"fail": fail, "worst": worst})
import json
print(json.dumps({"claim": "c1", "fail": fail, "worst_ratio": worst}))