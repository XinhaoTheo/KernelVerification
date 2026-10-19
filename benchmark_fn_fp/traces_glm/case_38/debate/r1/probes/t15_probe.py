import sys, torch, json
sys.path.insert(0, "/root/cases/case_111")
import kernel as K
dev = "cuda"
best = -1.0; best_desc = None; best_ratio_dw = -1.0
def trial(x, dy, w, desc):
    global best, best_desc, best_ratio_dw
    rstd = torch.rsqrt(x.float().square().mean(1) + 1e-5)
    ins = tuple(t.to(dev) for t in (x, w, dy, rstd))
    dx, dw = K.run(*ins)
    r = K.error_ratios((dx, dw), ins)
    if r["dx"] > best:
        best, best_desc = r["dx"], desc
    best_ratio_dw = max(best_ratio_dw, r["dw"])
g = torch.Generator("cpu").manual_seed(7)
M, N = 64, 512
# structured dominant rows
x = torch.full((M, N), 4.0).half(); dy = torch.full((M, N), 4.0).half(); w = torch.full((N,), 2.0).half()
trial(x, dy, w, "all x=+4 dy=4 w=2")
for seed in range(80):
    gg = torch.Generator("cpu").manual_seed(seed)
    # dominant random-sign rows, RMS=4 -> rstd~0.25; also mixed small rows for rstd~4
    x = (4.0*(2*(torch.rand(M, N, generator=gg) > 0.5).float() - 1)).half()
    dy = (4.0*(2*(torch.rand(M, N, generator=gg) > 0.5).float() - 1)).half()
    w = (1.0 + torch.rand(N, generator=gg)).half()  # |m| up to 8
    trial(x, dy, w, f"rand-sign seed {seed}")
    # small RMS rows (rstd~4) with dominant m
    x2 = (0.25*(2*(torch.rand(M, N, generator=gg) > 0.5).float() - 1)).half()
    trial(x2, dy, w, f"small-rms seed {seed}")
print(json.dumps({"trials": 161, "M": 64, "N": 512,
                  "max_dx_ratio": best, "argmax_case": best_desc,
                  "max_dw_ratio_same_trials": best_ratio_dw,
                  "claim_decided_by": "max_dx_ratio > 1 on legal dominant-row inputs"}))