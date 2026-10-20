import torch, json, sys
sys.path.insert(0, "/root/cases/case_99")
import kernel as K

vals = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
vals = vals[torch.isfinite(vals)]
m = vals.abs()
# outer-branch large magnitudes, up to 65504
sub = vals[m >= 1024]
sub = sub.cuda()
out = K.run(sub)
ref = torch.asinh(sub.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
i = err.argmax()
order = err.argsort(descending=True)
print(json.dumps({
    "metric": "kernel FP32 outer-branch output absolute error vs FP64 asinh, |x| in [1024, 65504]",
    "n_values": int(sub.numel()),
    "max_abs_err": float(err[i]),
    "tolerance": 1e-6,
    "worst_x": float(sub[i].to(torch.float64)),
    "worst_5": [[float(sub[j].to(torch.float64)), float(err[j])] for j in order[:5]],
    "n_exceed_1e-6": int((err > 1e-6).sum()),
    "exceeds_1e-6": bool((err > 1e-6).any()),
}))