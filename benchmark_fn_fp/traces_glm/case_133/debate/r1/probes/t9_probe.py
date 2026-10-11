import torch, json, sys
sys.path.insert(0, "/root/cases/case_133")
import kernel

dev = "cuda"
# large-magnitude legal FP16 values
vals = [65504.0, -65504.0, 4096.0, 1024.0, 512.0, 256.0, 128.0, 64.0, 32.0, 20.0, 25.0, 40.0, 100.0, 1000.0]
x = torch.tensor(vals, dtype=torch.float16, device=dev)
# also large-magnitude distribution like make_inputs
g = torch.Generator(device="cpu").manual_seed(0)
xr = (32.0 * torch.randn((4093,), generator=g)).to(torch.float16).to(dev)
xl = torch.randn((4093,), generator=torch.Generator(device="cpu").manual_seed(1).get_state() if False else None) if False else None

xall = torch.cat([x, xr])
out = kernel.run(xall)
ref = torch.sin(xall.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()

sel = err[:len(vals)]
res = {
    "tolerance": 1e-6,
    "n": int(xall.numel()),
    "n_violations_total": int((err > 1e-6).sum().item()),
    "n_violations_randn32": int((err[len(vals):] > 1e-6).sum().item()),
    "max_err_randn32": float(err[len(vals):].max().item()),
    "curated_vals": [float(v) for v in vals],
    "curated_errs": [float(e) for e in sel],
    "curated_violations": [bool(e > 1e-6) for e in sel],
    "worst_value_overall": float(xall[int(err.argmax())].item()),
    "max_err_overall": float(err.max().item()),
}
print(json.dumps(res))
