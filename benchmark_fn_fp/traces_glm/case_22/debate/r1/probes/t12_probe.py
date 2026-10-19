import torch, json, traceback, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_23/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
res = {}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
for C in [3, 48, 4]:
    try:
        x = torch.randn(2, C, 5, 5, device=dev)
        s = torch.randn(C, device=dev)
        out = k.scale_channels(x, s, C)
        ref = x * s.view(1, C, 1, 1)
        err = (out - ref).abs().max().item() if out.shape == ref.shape else None
        res[f"C{C}"] = {"raised": False, "max_abs_err_vs_nchw_ref": err}
    except Exception as e:
        res[f"C{C}"] = {"raised": True, "exc_type": type(e).__name__, "msg": str(e)[:200]}
print(json.dumps({"device": dev, "results": res}))