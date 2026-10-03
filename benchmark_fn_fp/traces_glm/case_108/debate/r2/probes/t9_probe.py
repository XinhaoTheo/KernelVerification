
import json, sys, torch, triton, triton.language as tl
sys.path.insert(0, "/root/cases/case_108")
import kernel as K

dev = "cuda"

@triton.jit
def _acc(A, B, C, N: tl.constexpr, BLK: tl.constexpr):
    i = tl.arange(0, BLK)
    m = i < N
    a = tl.load(A + i, mask=m)
    b = tl.load(B + i, mask=m)
    acc = tl.zeros((BLK,), tl.float32)
    acc += a * b
    tl.store(C + i, acc, mask=m)

def semantics_acc(n=4096, seed=0):
    g = torch.Generator("cpu").manual_seed(seed)
    a = ((torch.rand(n, generator=g) * 3 + 0.5)).half().to(dev)
    b = ((torch.rand(n, generator=g) * 3 + 0.5)).half().to(dev)
    c = torch.empty(n, dtype=torch.float32, device=dev)
    _acc[(1,)](a, b, c, n, triton.next_power_of_2(n),
              enable_fp_fusion=False)
    fprod = a.float() * b.float()
    h16 = fprod.half().float()
    return {
        "acc_vs_h16_max": float((c - h16).abs().max().item()),
        "acc_vs_f32_max": float((c - fprod).abs().max().item()),
    }

def dw_stress(m_, n_, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    x = (torch.randn((m_, n_), generator=g) * 0.25).clamp(-4, 4).half()
    rstd = torch.rsqrt(x.float().square().mean(1) + 1e-5)
    w = torch.full((n_,), 2.0, dtype=torch.float16)
    dy = (torch.randn((m_, n_), generator=g) * 4).clamp(-4, 4)
    dy = (dy - dy.mean(0, keepdim=True)).half()  # cancellation-heavy columns
    x, w, dy, rstd = (t.to(dev) for t in (x, w, dy, rstd))
    block = triton.next_power_of_2(n_)
    path = "row" if (block > 256 or m_ < 512) else "block"
    dx, dw = K.run(x, w, dy, rstd)
    r = K.error_ratios((dx, dw), (x, w, dy, rstd))
    # cancellation diagnostics: how small is dw target vs sum|t|
    h = (x.float() * rstd[:, None]).half()
    t = (dy * h).half()
    abs_terms = t.double().abs().sum(0)
    ref_dw = t.double().sum(0)
    return {
        "M": m_, "N": n_, "path": path, "dx_ratio": r["dx"], "dw_ratio": r["dw"],
        "min_abs_sum_over_ref_abs": float((abs_terms / ref_dw.abs().clamp_min(1e-30)).min().item()),
        "mean_abs_sum_over_ref_abs": float((abs_terms / ref_dw.abs().clamp_min(1e-30)).mean().item()),
        "all_finite": bool(torch.isfinite(dx).all().item() and torch.isfinite(dw).all().item()),
    }

out = {"semantics_acc": semantics_acc()}
for (m_, n_) in [(4096, 512), (4096, 128)]:
    for s in (1, 2, 3):
        r = dw_stress(m_, n_, s)
        out[f"stress_M{m_}_N{n_}_s{s}"] = r
print(json.dumps(out))
