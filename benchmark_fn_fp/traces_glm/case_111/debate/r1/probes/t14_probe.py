import sys, torch, json
sys.path.insert(0, "/root/cases/case_111")
import kernel as K
dev = "cuda"
# Diagnostic: is torch's (dy*weight) with fp16 inputs already H16-rounded?
g = torch.Generator("cpu").manual_seed(123)
dy = (2*torch.rand(10000, generator=g)).half()
w  = (2*torch.rand(10000, generator=g)).half()
torch_fp16_prod = (dy*w)
exact_then_round = (dy.double()*w.double()).to(torch.float16)
asym_mismatch_frac = float((torch_fp16_prod != exact_then_round).float().mean().item())
# Worst-case search: rstd ~4 (tiny RMS rows), |dy|=4, weight in [1,2] -> |m| up to 8
best = -1.0; best_seed = None; worst_ratio_dw = -1.0
for seed in range(80):
    g = torch.Generator("cpu").manual_seed(seed)
    M, N = 64, 512
    x = (0.03*torch.randn(M, N, generator=g)).half().clamp(-4, 4)
    rstd = torch.rsqrt(x.float().square().mean(1) + 1e-5)
    dy = (4.0*(2*(torch.rand(M, N, generator=g) > 0.5).float() - 1)).half()
    w = (1.0 + torch.rand(N, generator=g)).half()
    ins = tuple(t.to(dev) for t in (x, w, dy, rstd))
    dx, dw = K.run(*ins)
    r = K.error_ratios((dx, dw), ins)
    if r["dx"] > best:
        best, best_seed = r["dx"], seed
    worst_ratio_dw = max(worst_ratio_dw, r["dw"])
print(json.dumps({"asym_m_mismatch_fraction_torch_fp16mul_vs_rounded_exact": asym_mismatch_frac,
                  "trials": 80, "M": 64, "N": 512,
                  "max_dx_ratio": best, "argmax_seed": best_seed,
                  "max_dw_ratio_same_trials": worst_ratio_dw,
                  "claim_decided_by": "max_dx_ratio > 1 on legal adversarial inputs"}))