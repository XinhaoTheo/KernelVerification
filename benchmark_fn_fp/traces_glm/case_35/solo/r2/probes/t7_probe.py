import torch, sys, json
sys.path.insert(0, "/root/cases/case_108")
import kernel as K

dev = "cuda"
results = []
worst = {"dx": 0.0, "dw": 0.0}

def make(rows, cols, seed, extreme=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if extreme:
        # extreme magnitudes: |x|,|dy| up to 4, weight up to 2, RMS of each row in [0.25,4]
        x = (torch.rand((rows, cols), generator=g) * 8 - 4).half()
        # force RMS near 4: scale each row
        scale = torch.empty(rows).uniform_(0.25, 4.0, generator=g)
        x = (x.float() * (scale / x.float().square().mean(dim=1).add(1e-5).sqrt().clamp_min(1e-6))).clamp_(-4, 4).half()
        dy = (torch.rand((rows, cols), generator=g) * 8 - 4).half()
        weight = (torch.rand((cols,), generator=g) * 2).half()
    else:
        x = (2 * torch.rand((rows, cols), generator=g) - 1).half()
        dy = (2 * torch.rand((rows, cols), generator=g) - 1).half()
        weight = (0.5 + torch.rand((cols,), generator=g)).half()
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    return tuple(t.to(dev) for t in (x, weight, dy, rstd))

shapes = [
    # (M, N) covering row-kernel branch (block>256 or m<512) and block branch
    (1, 16), (1, 512), (7, 100), (17, 33), (511, 128), (512, 128),
    (512, 512), (512, 257), (1000, 300), (4096, 512), (4096, 16),
    (2048, 255), (513, 256), (511, 64), (3, 17),
]
fails = []
for idx, (m, n) in enumerate(shapes):
    for extreme in (False, True):
        inputs = make(m, n, seed=idx * 2 + int(extreme), extreme=extreme)
        x, weight, dy, rstd = inputs
        snapshots = [t.clone() for t in inputs]
        dx, dw = K.run(x, weight, dy, rstd)
        # preservation
        for t, s, name in zip(inputs, snapshots, ["x", "weight", "dy", "rstd"]):
            if not torch.equal(t, s):
                fails.append(f"input {name} modified at shape {(m,n)} ext={extreme}")
        # shapes/dtypes/finiteness
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
            fails.append(f"tolerance exceed at shape {(m,n)} ext={extreme}: {ratios}")

print(json.dumps({
    "metric": "error_ratios (ratio<=1 satisfies contract) over 14 shapes x benign/extreme, plus input-preservation, dtype/shape, finiteness checks",
    "shapes_tested": len(shapes) * 2,
    "worst_dx_ratio": worst["dx"],
    "worst_dw_ratio": worst["dw"],
    "failures": fails,
}))
