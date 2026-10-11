import torch, json, sys
sys.path.insert(0, "/root/cases/case_123")
import kernel

torch.manual_seed(0)
res = {}
for (H, W) in [(31, 33), (8, 8), (5, 9)]:
    x = (2*torch.rand((H, W), generator=torch.Generator().manual_seed(1)) - 1).half().cuda()
    actual = kernel.run(x, 4)
    ref = kernel.reference(x.cpu(), 4).cuda()
    err = (actual - ref).abs()
    # near-edge mask: within 4 of any border
    m = torch.zeros((H, W), dtype=torch.bool, device="cuda")
    m[:4, :] = m[-4:, :] = True; m[:, :4] = m[:, -4:] = True
    res[f"{H}x{W}"] = {
        "max_err_near_edge": err[m].max().item(),
        "max_err_interior": err[~m].max().item() if (~m).any() else None,
        "n_over_tol": (err > 2e-5).sum().item(),
        "argmax": torch.unravel_index(err.argmax(), (H, W)),
    }
print(json.dumps(res))