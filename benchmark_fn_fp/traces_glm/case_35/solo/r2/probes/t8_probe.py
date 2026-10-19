import torch, sys, json
sys.path.insert(0, "/root/cases/case_108")
import kernel as K

dev = "cuda"

def make(rows, cols, seed, extreme=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if extreme:
        x = (torch.rand((rows, cols), generator=g) * 8 - 4).half()
        target_rms = 0.25 + (4.0 - 0.25) * torch.rand((rows,), generator=g)
        cur_rms = x.float().square().mean(dim=1).add(1e-5).sqrt()
        x = (x.float() * (target_rms / cur_rms)[:, None]).clamp_(-4, 4).half()
        dy = (torch.rand((rows, cols), generator=g) * 8 - 4).half()
        weight = (torch.rand((cols,), generator=g) * 2).half()
    else:
        x = (2 * torch.rand((rows, cols), generator=g) - 1).half()
        dy = (2 * torch.rand((rows, cols), generator=g) - 1).half()
        weight = (0.5 + torch.rand((cols,), generator=g)).half()
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    return tuple(t.to(dev) for t in (x, weight, dy, rstd))

shapes = [
    (1, 16), (1, 512), (7, 100), (17, 33), (511, 128), (512, 128),
    (512, 512), (512, 257), (1000, 300), (4096, 512), (4096, 16),
    (2048, 255), (513, 256), (511, 64), (3, 17),
]
fails = []
worst = {"dx": 0.0, "dw": 0.0}
for idx, (m, n) in enumerate(shapes):
    for extreme in (False, True):
        inputs = make(m, n, seed=idx * 2 + int(extreme), extreme=extreme)
        x, weight, dy, rstd = inputs
        snaps = [t.clone() for t in inputs]
        # domain checks
        assert torch.isfinite(x).all() and x.float().abs().max() <= 4
        assert dy.float().abs().max() <= 4 and weight.float().abs().max() <= 2
        rms = x.float().square().mean(dim=1).add(1e-5).sqrt()
        assert rms.min() >= 0.25 - 1e-3 and rms.max() <= 4 + 1e-3, (m, n, rms.min(), rms.max())
        dx, dw = K.run(x, weight, dy, rstd)
        for t, s, name in zip(inputs, snaps, ["x", "weight", "dy", "rstd"]):
            if not torch.equal(t, s):
                fails.append(f"input {name} modified at shape {(m,n)} ext={extreme}")
        if dx.dtype != torch.float16 or dx.shape != (m, n):
            fails.append(f"dx shape/dtype at {(m,n)}")
        if dw.dtype != torch.float32 or dw.shape != (n,):
            fails.append(f"dw shape/dtype at {(m,n)}")
        if not torch.isfinite(dx).all() or not torch.isfinite(dw).all():
            fails.append(f"nonfinite output at {(m,n)} ext={extreme}")
        ratios = K.error_ratios((dx, dw), inputs)
        worst["dx"] = max(worst["dx"], ratios["dx"])
        worst["dw"] = max(worst["dw"], ratios["dw"])
        if ratios["dx"] > 1.0 or ratios["dw"] > 1.0:
            fails.append(f"tolerance exceed at {(m,n)} ext={extreme}: {ratios}")

print(json.dumps({
    "metric": "error_ratios (ratio<=1 satisfies contract) over 15 shapes x benign/extreme, plus input-preservation, dtype/shape, finiteness checks",
    "cases_tested": len(shapes) * 2,
    "worst_dx_ratio": worst["dx"],
    "worst_dw_ratio": worst["dw"],
    "failures": fails,
}))