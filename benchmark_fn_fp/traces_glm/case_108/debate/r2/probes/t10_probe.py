
import json, sys, torch, triton, triton.language as tl
sys.path.insert(0, "/root/cases/case_108")
import kernel as K

dev = "cuda"

@triton.jit
def _mul(A, B, C, N: tl.constexpr, BLK: tl.constexpr):
    i = tl.arange(0, BLK)
    m = i < N
    a = tl.load(A + i, mask=m)
    b = tl.load(B + i, mask=m)
    tl.store(C + i, (a * b).to(tl.float32), mask=m)

def semantics_mul(n=4096, seed=0):
    g = torch.Generator("cpu").manual_seed(seed)
    a = ((torch.rand(n, generator=g) * 3 + 0.5)).half().to(dev)
    b = ((torch.rand(n, generator=g) * 1.5 + 0.5)).half().to(dev)
    c = torch.empty(n, dtype=torch.float32, device=dev)
    _mul[(1,)](a, b, c, n, triton.next_power_of_2(n), enable_fp_fusion=False)
    fprod = a.float() * b.float()
    h16 = fprod.half().float()
    return {
        "mul_vs_h16_max": float((c - h16).abs().max().item()),
        "mul_vs_f32_max": float((c - fprod).abs().max().item()),
    }

def dx_stress(m_, n_, rms_scale, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    x = (torch.randn((m_, n_), generator=g) * rms_scale).clamp(-4, 4).half()
    rstd = torch.rsqrt(x.float().square().mean(1) + 1e-5)
    w = torch.full((n_,), 2.0, dtype=torch.float16)
    dy = (torch.randn((m_, n_), generator=g) * 4).clamp(-4, 4).half()
    x, w, dy, rstd = (t.to(dev) for t in (x, w, dy, rstd))
    block = triton.next_power_of_2(n_)
    path = "row" if (block > 256 or m_ < 512) else "block"
    dx, dw = K.run(x, w, dy, rstd)
    r = K.error_ratios((dx, dw), (x, w, dy, rstd))
    return {"M": m_, "N": n_, "path": path, "rstd_min": float(rstd.min()), "rstd_max": float(rstd.max()),
            "dx_ratio": r["dx"], "dw_ratio": r["dw"],
            "dx_abs_max": float(dx.abs().max().item()),
            "all_finite": bool(torch.isfinite(dx).all().item() and torch.isfinite(dw).all().item())}

out = {"semantics_mul": semantics_mul()}
# rstd extreme ~4: rows RMS ~0.25
for (m_, n_) in [(4096, 512), (4096, 100)]:
    for s in (1, 2):
        out[f"dx_hi_rstd_M{m_}_N{n_}_s{s}"] = dx_stress(m_, n_, 0.25, s)
# rstd extreme ~0.25: rows RMS ~4 (|x|=4 constant)
out["dx_lo_rstd_M4096_N100"] = dx_stress(4096, 100, None, 3) if False else None
del out["dx_lo_rstd_M4096_N100"]
def dx_const(m_, n_, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    x = (torch.sign(torch.randn((m_, n_), generator=g)) * 4).half()
    rstd = torch.rsqrt(x.float().square().mean(1) + 1e-5)
    w = torch.full((n_,), 2.0, dtype=torch.float16)
    dy = (torch.sign(torch.randn((m_, n_), generator=g)) * 4).half()
    x, w, dy, rstd = (t.to(dev) for t in (x, w, dy, rstd))
    block = triton.next_power_of_2(n_)
    path = "row" if (block > 256 or m_ < 512) else "block"
    dx, dw = K.run(x, w, dy, rstd)
    r = K.error_ratios((dx, dw), (x, w, dy, rstd))
    return {"M": m_, "N": n_, "path": path, "rstd_min": float(rstd.min()), "rstd_max": float(rstd.max()),
            "dx_ratio": r["dx"], "dw_ratio": r["dw"], "dx_abs_max": float(dx.abs().max().item()),
            "all_finite": bool(torch.isfinite(dx).all().item() and torch.isfinite(dw).all().item())}
out["dx_lo_rstd_M4096_N100"] = dx_const(4096, 100, 3)
print(json.dumps(out))
