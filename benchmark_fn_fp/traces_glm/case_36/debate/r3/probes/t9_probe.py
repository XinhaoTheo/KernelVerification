import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_109/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
out = {}
for name, M, N in [("const_same_sign", 4096, 512)]:
    x = torch.full((M, N), 2.0, dtype=torch.float16, device=dev)
    dy = torch.full((M, N), 4.0, dtype=torch.float16, device=dev)
    weight = torch.full((N,), 2.0, dtype=torch.float16, device=dev)
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    dx, dw = k.run(x, weight, dy, rstd)
    h = (x.float() * rstd[:, None]).to(torch.float16)
    tH = (dy * h)  # H16-rounded product
    tF = (dy.float() * h.float())  # product promoted to F32, no H16 rounding
    dwA = tH.double().sum(0)
    dwB = tF.double().sum(0)
    lim = 1e-5 + 1e-5 * tH.double().abs().sum(0)
    errA = ((dw.double() - dwA).abs() / lim).max().item()
    errB = ((dw.double() - dwB).abs() / lim).max().item()
    errA_limB = ((dw.double() - dwA).abs() / (1e-5 + 1e-5 * tF.double().abs().sum(0))).max().item()
    out[name] = {
        "dw_vs_H16terms_ratio": errA,
        "dw_vs_F32promoted_terms_ratio": errB,
        "dw_vs_H16terms_ratio_using_F32_abs_sum": errA_limB,
        "terms_identical": bool((tH == tF).all().item()),
        "dw[0]": dw[0].item(), "dwA[0]": dwA[0].item(), "dwB[0]": dwB[0].item(),
        "sum_abs_t": tH.double().abs().sum(0)[0].item(),
    }
# also a random same-sign case for robustness (bias direction may vary)
g = torch.Generator(device="cpu").manual_seed(1)
M, N = 4096, 512
x = (0.5 + 1.5 * torch.rand((M, N), generator=g)).half().to(dev)
dy = (2 * torch.rand((M, N), generator=g) + 2).half().to(dev)
weight = (0.5 + 1.5 * torch.rand((N,), generator=g)).half().to(dev)
rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
dx, dw = k.run(x, weight, dy, rstd)
h = (x.float() * rstd[:, None]).to(torch.float16)
tH = (dy * h); tF = (dy.float() * h.float())
dwA = tH.double().sum(0); dwB = tF.double().sum(0)
limA = 1e-5 + 1e-5 * tH.double().abs().sum(0)
limB = 1e-5 + 1e-5 * tF.double().abs().sum(0)
out["rand_same_sign"] = {
    "dw_vs_H16terms_ratio": ((dw.double() - dwA).abs() / limA).max().item(),
    "dw_vs_F32promoted_terms_ratio": ((dw.double() - dwB).abs() / limB).max().item(),
    "n_terms_differ": (tH != tF).sum().item(),
    "max_half_ulp_gap": (tH.double() - tF.double()).abs().max().item(),
}
print(json.dumps(out))