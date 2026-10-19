import sys, json, torch
sys.path.insert(0, "/root/cases/case_109")
import kernel
dev = "cuda"

def make(M, N, seed, mode):
    g = torch.Generator("cpu").manual_seed(seed)
    if mode == "const":
        x = torch.full((M, N), 0.25); dy = torch.full((M, N), 4.0); w = torch.full((N,), 2.0)
    elif mode == "randpos":
        x = 0.05 + 0.40 * torch.rand((M, N), generator=g)
        dy = 2.0 + 2.0 * torch.rand((M, N), generator=g)
        w = 1.0 + torch.rand((N,), generator=g)
    else:  # mixedexp: positive values spanning exponents, row-normalized to RMS 0.30
        e = torch.randint(-8, 2, (M, N), generator=g).float()
        x = (2.0 ** e) * (0.5 + 0.5 * torch.rand((M, N), generator=g))
        rms = x.square().mean(1, keepdim=True).sqrt().clamp(min=1e-6)
        x = x * (0.30 / rms)
    xh = x.half(); wh = w.half(); dyh = dy.half()
    rstd = torch.rsqrt(xh.float().square().mean(1) + 1e-5).float()
    return tuple(t.to(dev) for t in (xh, wh, dyh, rstd))

def trial(M, N, seed, mode):
    xh, wh, dyh, rstd = make(M, N, seed, mode)
    T = (xh, wh, dyh, rstd)
    dx, dw = kernel.run(*T)
    rat = kernel.error_ratios((dx, dw), T)
    # independent FP64 real-arithmetic dw target: sum of individually H16-rounded t terms
    h = (xh.float() * rstd[:, None]).to(torch.float16)
    t = (dyh * h)
    t64 = t.double()
    ref_dw = t64.sum(0)
    lim = 1e-5 + 1e-5 * t64.abs().sum(0)
    dwr = float(((dw.double() - ref_dw).abs() / lim).max())
    return dict(M=M, N=N, mode=mode, seed=seed,
                dx_ratio=round(rat["dx"], 4), dw_ratio=round(rat["dw"], 4),
                dw_ratio_direct=round(dwr, 4),
                finite=bool(dx.isfinite().all() and dw.isfinite().all()))

res = []
for N in (128, 300, 512):          # 128: block path; 300/512: row path at M=4096
    for mode in ("const", "randpos", "mixedexp"):
        for seed in range(5):
            res.append(trial(4096, N, seed, mode))
# shape/coverage sweep incl. non-divisible M and edge shapes
for (M, N) in ((1, 16), (7, 17), (511, 128), (512, 200), (517, 257), (4001, 300), (4095, 512), (4096, 16)):
    for seed in range(3):
        res.append(trial(M, N, seed, "randpos"))

# input-preservation check on one large run
xh, wh, dyh, rstd = make(4096, 512, 99, "randpos")
c = [t.clone() for t in (xh, wh, dyh, rstd)]
kernel.run(xh, wh, dyh, rstd)
inputs_preserved = all(torch.equal(a, b) for a, b in zip(c, (xh, wh, dyh, rstd)))

out = dict(
    max_dw_ratio_direct=max(r["dw_ratio_direct"] for r in res),
    max_dw_ratio_error_ratios=max(r["dw_ratio"] for r in res),
    max_dx_ratio=max(r["dx_ratio"] for r in res),
    all_finite=all(r["finite"] for r in res),
    worst_dw_case=max(res, key=lambda r: r["dw_ratio_direct"]),
    worst_dx_case=max(res, key=lambda r: r["dx_ratio"]),
    inputs_preserved=bool(inputs_preserved),
    n_trials=len(res),
)
print(json.dumps(out))