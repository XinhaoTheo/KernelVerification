import torch, json, sys
sys.path.insert(0, "/root/cases/case_123")
import kernel

res = {}
for (H, W) in [(31, 33), (8, 8), (5, 9)]:
    x = (2*torch.rand((H, W), generator=torch.Generator().manual_seed(1)) - 1).half().cuda()
    actual = kernel.run(x, 4)
    ref = kernel.reference(x.cpu(), 4).cuda()
    err = (actual - ref).abs()
    m = torch.zeros((H, W), dtype=torch.bool, device="cuda")
    m[:4, :] = True; m[-4:, :] = True; m[:, :4] = True; m[:, -4:] = True
    am = torch.unravel_index(err.argmax(), (H, W))
    res[f"{H}x{W}"] = {
        "max_err_near_edge": err[m].max().item(),
        "max_err_interior": err[~m].max().item() if (~m).any() else None,
        "n_over_tol": (err > 2e-5).sum().item(),
        "argmax": [int(am[0]), int(am[1])],
    }
print(json.dumps(res))