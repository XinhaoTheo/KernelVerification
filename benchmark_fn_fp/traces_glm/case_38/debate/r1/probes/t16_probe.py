import sys, torch, json
sys.path.insert(0, "/root/cases/case_111")
import kernel as K
dev = "cuda"
best = -1.0; best_desc = None; best_dx = -1.0
M = 4096
for N in (256, 512):  # N=256 exercises _block_backward path, N=512 the row path
    for seed in range(40):
        g = torch.Generator("cpu").manual_seed(seed)
        x = (0.25*(2*(torch.rand(M, N, generator=g) > 0.5).float() - 1)).half()  # RMS=0.25 -> rstd~4, h~1
        rstd = torch.rsqrt(x.float().square().mean(1) + 1e-5)
        dy = (4.0*(2*(torch.rand(M, N, generator=g) > 0.5).float() - 1)).half()
        w = (1.0 + torch.rand(N, generator=g)).half()
        # engineered per-column cancellation: flip dy signs by row parity
        dy_cancel = dy.clone(); dy_cancel[::2] = -dy_cancel[::2]
        for tag, d in (("rand", dy), ("cancel", dy_cancel)):
            ins = tuple(t.to(dev) for t in (x, w, d, rstd))
            dx, dw = K.run(*ins)
            r = K.error_ratios((dx, dw), ins)
            if r["dw"] > best:
                best, best_desc = r["dw"], f"N={N} seed={seed} {tag}"
            best_dx = max(best_dx, r["dx"])
print(json.dumps({"trials": 80, "M": 4096, "N_tested": [256, 512],
                  "max_dw_ratio": best, "argmax_case": best_desc,
                  "max_dx_ratio_same_trials": best_dx,
                  "claim_decided_by": "max_dw_ratio > 1 on cancellation-heavy legal columns"}))