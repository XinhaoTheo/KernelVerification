import importlib.util, torch, triton, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_109/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(3)
M, N = 512, 200  # block path selected by run(); non-power-of-two N
x = ((2 * torch.rand((M, N), generator=g) - 1) * 4).half().to(dev)
dy = ((2 * torch.rand((M, N), generator=g) - 1) * 4).half().to(dev)
weight = ((0.5 + 1.5 * torch.rand((N,), generator=g))).half().to(dev)
rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
# run() -> block kernel
dx_blk, dw_blk = k.run(x, weight, dy, rstd)
# invoke row kernel directly on identical inputs
block = triton.next_power_of_2(N)
programs = min(torch.cuda.get_device_properties(x.device).multi_processor_count, M)
partial = torch.empty((programs, N), dtype=torch.float32, device=dev)
dx_row = torch.empty_like(dy)
k._row_backward[(programs,)](dy, dx_row, x, weight, rstd, partial, M, N,
    ROWS=triton.cdiv(M, programs), B=block, num_warps=4, enable_fp_fusion=False)
dw_row = partial.sum(dim=0)
dx_diff = (dx_blk.double() - dx_row.double()).abs().max().item()
dw_diff = (dw_blk.double() - dw_row.double()).abs().max().item()
r = k.error_ratios((dx_blk, dw_blk), (x, weight, dy, rstd))
r_row = k.error_ratios((dx_row, dw_row), (x, weight, dy, rstd))
# extremes check: |x|=4, |dy|=4, weight=2, both kernels
M2 = 511
x2 = torch.full((M2, N), 4.0, dtype=torch.float16, device=dev)
dy2 = torch.full((M2, N), 4.0, dtype=torch.float16, device=dev)
w2 = torch.full((N,), 2.0, dtype=torch.float16, device=dev)
rstd2 = torch.rsqrt(x2.float().square().mean(dim=1) + 1e-5)
dx2, dw2 = k.run(x2, w2, dy2, rstd2)
r2 = k.error_ratios((dx2, dw2), (x2, w2, dy2, rstd2))
out = {
    "M": M, "N": N,
    "dx_path_max_abs_diff": dx_diff,
    "dw_path_max_abs_diff": dw_diff,
    "block_path_error_ratios": r,
    "row_path_error_ratios": r_row,
    "extremes_M511_error_ratios": r2,
    "dx_ratio_over_1": bool(max(r["dx"], r_row["dx"], r2["dx"]) > 1.0),
    "dw_ratio_over_1": bool(max(r["dw"], r_row["dw"], r2["dw"]) > 1.0),
}
print(json.dumps(out))