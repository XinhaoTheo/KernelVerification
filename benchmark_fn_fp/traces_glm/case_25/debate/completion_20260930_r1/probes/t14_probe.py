import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_25/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
torch.manual_seed(0)
x = torch.randn(4096, device="cuda")
y = mod.requantize(x)
# contract reference: per-1024-block scale, q = round-to-nearest
xflat = x.reshape(-1)
n = xflat.numel(); B = 1024
yref = torch.empty_like(xflat)
for i in range(0, n, B):
    b = xflat[i:i+B]
    absmax = b.abs().max()
    scale = 1.0 if absmax == 0 else absmax / 127.0
    q = torch.round(b / scale)
    yref[i:i+B] = q * scale
diff = (y - yref).abs()
scale = xflat.abs().max() / 127.0
mismatch = (diff > 1e-6 * scale).sum().item()
print(json.dumps({"metric": "elementwise error vs round-to-nearest reference",
 "reason": "contract requires q=round(x/scale) to nearest; kernel uses floor",
 "max_abs_err": diff.max().item(), "n_mismatched": mismatch, "n_total": n,
 "frac_mismatched": mismatch / n, "max_err_in_quant_steps": (diff.max() / scale).item()}))