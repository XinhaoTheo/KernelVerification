import torch, json
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':
    print(json.dumps({"error": "no cuda"}))
else:
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_04/kernel.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    def ref(X, eps):
        X32 = X.float()
        ms = (X32 * X32).sum(-1) / X32.shape[-1]
        rstd = torch.rsqrt(ms + eps)
        return X32 * rstd.unsqueeze(-1), rstd

    results = {}
    cases = {
        "fp32_randn": torch.randn(64, 256, device=dev),
        "fp32_small": torch.randn(64, 256, device=dev) * 1e-6,
        "fp32_tiny": torch.randn(64, 256, device=dev) * 1e-30,
        "bf16_roundtrip": torch.randn(64, 256, device=dev).bfloat16().float(),
        "near_zero_row": None,
    }
    nz = torch.randn(64, 256, device=dev); nz[10] = 0.0; nz[20] = 1e-40
    cases["near_zero_row"] = nz
    eps = 1e-6
    for name, X in cases.items():
        Y, R = mod.rms_norm_forward(X, eps), None
        Yk = Y; Yr, Rr = ref(X, eps)
        # rstd not returned by wrapper; compare y only
        dy = (Yk - Yr)
        results[name] = {
            "max_abs_err": dy.abs().max().item(),
            "max_rel_err": (dy.abs() / (Yr.abs() + 1e-12)).max().item(),
            "finite": bool(torch.isfinite(Yk).all().item()),
        }
    # larger sizes
    for n in [7, 128, 1024, 4096]:
        X = torch.randn(8, n, device=dev)
        Yr, _ = ref(X, eps)
        Yk = mod.rms_norm_forward(X, eps)
        results[f"n_{n}"] = {"max_abs_err": (Yk - Yr).abs().max().item()}
    print(json.dumps(results))