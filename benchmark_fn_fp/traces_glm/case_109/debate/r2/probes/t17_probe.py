import sys, json, torch
sys.path.insert(0, "/root/cases/case_109")
import kernel
dev = "cuda"

def make(M, N, seed, mode):
    g = torch.Generator("cpu").manual_seed(seed)
    if mode == "dom16":   # N=16, one element 4 per row -> rstd~1, max amplification r^3*x/N
        x = 0.05 * torch.rand((M, N), generator=g); x[:, 0] = 4.0
        dy = torch.full((M, N), 4.0); w = torch.full((N,), 2.0)
    elif mode == "dom512":  # N=512, one element 4, rest ~0.2 -> rstd~3.8, huge sum(m*x)
        x = 0.19 + 0.02 * torch.rand((M, N), generator=g); x[:, 0] = 4.0
        dy = torch.full((M, N), 4.0); w = torch.full((N,), 2.0)
    elif mode == "maxsum":  # N=512 or 16: all |x|=4 -> rstd=0.25, sum(m*x)=32*N (16384 at N=512)
        x = torch.full((M, N), 4.0); dy = torch.full((M, N), 4.0); w = torch.full((N,), 2.0)
    elif mode == "mixsign":  # large |m*x| with random signs -> cancellation in the row sum
        x = 4.0 * (2 * torch.rand((M, N), generator=g) - 1)
        dy = 4.0 * (2 * torch.rand((M, N), generator=g) - 1)
        w = 2.0 * (2 * torch.rand((N,), generator=g) - 1)
    else:  # rand over full domain
        x = 2 * torch.rand((M, N), generator=g) - 1
        dy = 2 * torch.rand((M, N), generator=g) - 1
        w = 0.5 + 1.5 * torch.rand((N,), generator=g)
    xh = x.half(); dyh = dy.half(); wh = w.half()
    # keep row RMS in [0.25, 4]
    rms = xh.float().square().mean(1).sqrt()
    scale = torch.ones_like(rms)
    scale[rms < 0.25] = 0.3 / rms[rms < 0.25]
    scale[rms > 4.0] = 3.5 / rms[rms > 4.0]
    xh = (xh.float() * scale[:, None]).half()
    rstd = torch.rsqrt(xh.float().square().mean(1) + 1e-5).float()
    return tuple(t.to(dev) for t in (xh, wh, dyh, rstd)), xh, wh, dyh

def trial(M, N, seed, mode):
    T, xh, wh, dyh = make(M, N, seed, mode)
    dx, dw = kernel.run(*T)
    rat = kernel.error_ratios((dx, dw), T)
    rstd = T[3]
    return dict(M=M, N=N, mode=mode, seed=seed,
                rstd_min=float(rstd.min()), rstd_max=float(rstd.max()),
                dx_ratio=round(rat["dx"], 4), dw_ratio=round(rat["dw"], 4),
                max_abs_dx=float(dx.float().abs().max()),
                finite=bool(dx.isfinite().all() and dw.isfinite().all()))

res = []
for mode, M, Ns in (("dom16", 4096, [16]), ("dom512", 4096, [512]),
                    ("maxsum", 4096, [16, 512]), ("mixsign", 4096, [16, 512, 257]),
                    ("rand", 4096, [16, 17, 257, 512]), ("rand", 517, [128]),
                    ("rand", 1, [16, 512]), ("rand", 511, [512])):
    for N in Ns:
        for seed in range(8):
            res.append(trial(M, N, seed, mode))

out = dict(
    max_dx_ratio=max(r["dx_ratio"] for r in res),
    max_dw_ratio=max(r["dw_ratio"] for r in res),
    all_finite=all(r["finite"] for r in res),
    max_abs_dx_seen=max(r["max_abs_dx"] for r in res),
    rstd_range=[min(r["rstd_min"] for r in res), max(r["rstd_max"] for r in res)],
    worst_dx_case=max(res, key=lambda r: r["dx_ratio"]),
    n_trials=len(res),
)
print(json.dumps(out))