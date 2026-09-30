import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_86/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, w = k.make_inputs("cuda")
out = k.run(x, w)
print(json.dumps({"finite": bool(torch.isfinite(out).all().item()), "nan_count": int(torch.isnan(out).sum().item()), "inf_count": int(torch.isinf(out).sum().item()), "num_outputs": out.numel(), "min_scale_rowmax_abs": float(w.abs().max(dim=1).values.min().item())}))