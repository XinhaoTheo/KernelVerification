import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_109/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(7)
M, N = 4096, 512
x = ((2 * torch.rand((M, N), generator=g) - 1) * 4).half().to(dev)  # |x|<=4, RMS in range
rms = x.float().square().mean(dim=1)
print("rms range", rms.min().item(), rms.max().item())
rstd = torch.rsqrt(rms + 1e-5)
weight = ((0.5 + 1.5 * torch.rand((N,), generator=g))).half().to(dev)
# heavy cancellation in dw: dy sign per column alternates; magnitudes max
dy = torch.empty((M, N), dtype=torch.float16, device=dev)
signs = torch.where(torch.arange(M) % 2 == 0, 1.0, -1.0)
for j in range(N):
    dy[:, j] = (signs * 4.0).half()
dx, dw = k.run(x, weight, dy, rstd)
h = (x.float() * rstd[:, None]).to(torch.float16)
tH = (dy * h)  # H16 product
tF = (dy.float() * h.float())  # F32 promoted product
dwA = tH.double().sum(0)  # exact target under H16 semantics
dwB = tF.double().sum(0)  # exact target under F32-promoted semantics
limA = 1e-5 + 1e-5 * tH.double().abs().sum(0)
errA = ((dw.double() - dwA).abs() / limA).max().item()
errB = ((dw.double() - dwB).abs() / limB).max().item() if False else ((dw.double() - dwB).abs() / (1e-5 + 1e-5 * tF.double().abs().sum(0))).max().item()
# semantics-matched accumulation error = min of the two
acc_err = min(errA, errB)
out = {
    "M": M, "N": N,
    "dw_vs_H16_target_ratio": errA,
    "dw_vs_F32_target_ratio": errB,
    "semantics_matched_accum_error_ratio": acc_err,
    "sum_abs_t_min": tH.double().abs().sum(0).min().item(),
    "sum_abs_t_max": tH.double().abs().sum(0).max().item(),
    "dw_abs_target_range": [dwA.abs().min().item(), dwA.abs().max().item()],
    "ratio_over_1": bool(acc_err > 1.0),
}
print(json.dumps(out))