import torch, json, sys
sys.path.insert(0, "/root/cases/case_123")
import kernel

res = {}
for (H, W) in [(31, 33), (16, 16)]:
    x = (2*torch.rand((H, W), generator=torch.Generator().manual_seed(2)) - 1).half().cuda()
    actual = kernel.run(x, 8)
    ref = kernel.reference(x.cpu(), 8).cuda()
    err = (actual - ref).abs()
    m = torch.zeros((H, W), dtype=torch.bool, device="cuda")
    m[:4, :] = m[-4:, :] = True; m[:, :4] = m[:, -4:] = True
    res[f"{H}x{W}"] = {
        "max_err_near_edge": err[m].max().item(),
        "max_err_interior": err[~m].max().item(),
        "n_over_tol": (err > 2e-5).sum().item(),
    }
print(json.dumps(res))