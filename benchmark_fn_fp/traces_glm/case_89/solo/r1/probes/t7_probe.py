import torch, json, sys
sys.path.insert(0, "/root/cases/case_162")
import kernel

dev = "cuda"
tol_a, tol_r = 1e-4, 1e-6
results = []
worst = 0.0
worst_desc = ""

def check(desc, values):
    global worst, worst_desc
    before = values.clone()
    out = kernel.run(values)
    exact = kernel.reference(values.cpu()).to(dev)
    err = (out.double() - exact)
    tol = tol_a + tol_r * exact.abs()
    viol = (err.abs() > tol)
    maxratio = (err.abs() / tol).max().item()
    finite = bool(torch.isfinite(out).all())
    shape_ok = tuple(out.shape) == (values.shape[0], values.shape[1]//2 + 1, 2) and out.dtype == torch.float32 and out.is_contiguous()
    preserv = torch.equal(before, values)
    sep = out.data_ptr() != values.data_ptr()
    if maxratio > worst:
        worst, worst_desc = maxratio, desc
    results.append(dict(desc=desc, max_ratio=maxratio, violations=int(viol.sum()), finite=finite, shape_ok=shape_ok, input_preserved=preserv, separate_storage=sep))

g = torch.Generator(device="cpu").manual_seed(0)
for N in (64, 128, 256):
    for B in (1, 7, 256):
        v = torch.randn((B, N), generator=g).to(torch.float16).to(dev)
        check(f"rand B={B} N={N}", v)
for N in (64, 128, 256):
    B = 8
    check(f"ones N={N}", torch.ones((B, N), dtype=torch.float16, device=dev))
    check(f"fp16max N={N}", torch.full((B, N), 65504.0, dtype=torch.float16, device=dev))
    check(f"fp16min N={N}", torch.full((B, N), -65504.0, dtype=torch.float16, device=dev))
    check(f"negzero N={N}", torch.full((B, N), -0.0, dtype=torch.float16, device=dev))
    check(f"subnormal N={N}", torch.full((B, N), 6e-8, dtype=torch.float16, device=dev))
    v = torch.zeros((B, N), dtype=torch.float16, device=dev)
    for i in range(B):
        v[i, i * 7 % N] = 1.0
    check(f"impulse N={N}", v)
    v = torch.tensor([[((-1.0)**j) * 1000.0 for j in range(N)]] * B, dtype=torch.float16, device=dev)
    check(f"nyquist N={N}", v)
    v = torch.randn((B, N), generator=g).to(torch.float16)
    v = (v * 1e4).to(torch.float16)  # stays finite
    assert torch.isfinite(v).all()
    check(f"bigrand N={N}", v.to(dev))
    v = (torch.randn((B, N), generator=g) * 1e-7).to(torch.float16)
    assert torch.isfinite(v).all()
    check(f"tinyrand N={N}", v.to(dev))
    # extreme mix: alternating +/- max
    v = torch.tensor([[((-1.0)**j) * 65504.0 for j in range(N)]] * B, dtype=torch.float16, device=dev)
    check(f"altmax N={N}", v)

print(json.dumps(dict(results=results, worst_ratio=worst, worst_case=worst_desc, any_violation=any(r["violations"]>0 or not r["finite"] or not r["shape_ok"] or not r["input_preserved"] or not r["separate_storage"] for r in results)), indent=1))
