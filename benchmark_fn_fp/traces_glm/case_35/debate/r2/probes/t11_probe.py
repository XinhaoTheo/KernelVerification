
import json, sys, torch, triton
sys.path.insert(0, "/root/cases/case_108")
import kernel as K

dev = "cuda"

def block_stress(m_, n_, mode, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    if mode == "hi_rstd":   # rows RMS ~0.25 -> rstd ~4
        x = (torch.randn((m_, n_), generator=g) * 0.25).clamp(-4, 4).half()
    else:                  # rows RMS = 4 -> rstd ~0.25
        x = (torch.sign(torch.randn((m_, n_), generator=g)) * 4).half()
    rstd = torch.rsqrt(x.float().square().mean(1) + 1e-5)
    w = torch.full((n_,), 2.0, dtype=torch.float16)
    dy = (torch.randn((m_, n_), generator=g) * 4).clamp(-4, 4).half()
    x, w, dy, rstd = (t.to(dev) for t in (x, w, dy, rstd))
    block = triton.next_power_of_2(n_)
    path = "row" if (block > 256 or m_ < 512) else "block"
    assert path == "block", (m_, n_, block)
    dx, dw = K.run(x, w, dy, rstd)
    r = K.error_ratios((dx, dw), (x, w, dy, rstd))
    return {"M": m_, "N": n_, "mode": mode, "path": path,
            "rstd_min": float(rstd.min()), "rstd_max": float(rstd.max()),
            "dx_ratio": r["dx"], "dw_ratio": r["dw"],
            "dx_abs_max": float(dx.abs().max().item()),
            "all_finite": bool(torch.isfinite(dx).all().item() and torch.isfinite(dw).all().item())}

out = {}
for (m_, n_) in [(4096, 100), (4096, 128), (4096, 256), (512, 100), (513, 100)]:
    for mode in ("hi_rstd", "lo_rstd"):
        for s in (1, 2):
            out[f"blk_M{m_}_N{n_}_{mode}_s{s}"] = block_stress(m_, n_, mode, s)
print(json.dumps(out))
