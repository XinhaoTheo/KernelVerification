
import sys, json, torch
sys.path.insert(0, "/root/cases/case_166")
from kernel import run

# all 65536 FP16 bit patterns, keep finite values > -1 (exclude both -inf-ish and -1... values <= -1)
bits = torch.arange(65536, dtype=torch.int32)
x_all = bits.to(torch.uint16).view(torch.float16)
mask = torch.isfinite(x_all) & (x_all > -1.0)
x = x_all[mask].to(torch.float16)
# signed zeros collapse to one representative of each sign; keep all (both zeros legal)
print("legal count:", x.numel(), file=sys.stderr)
xs = x.to("cuda")
out = run(xs)
ref = torch.log1p(x.to(torch.float64).cpu())
err = (out.cpu().to(torch.float64) - ref).abs()
maxerr = err.max().item()
i = int(err.argmax())
# worst region: values with mantissa near 1.65
u = 1.0 + x.to(torch.float32).cpu()
enc = u.view(torch.uint32) if hasattr(u, 'view') else None
# top-5 worst
top = torch.topk(err, 5)
print(json.dumps({
    "metric": "max abs error over exhaustive legal FP16 domain vs FP64 log1p",
    "n_legal": int(x.numel()),
    "max_abs_err": maxerr,
    "tolerance": 1e-6,
    "exceeds_1e-6": bool(maxerr > 1e-6),
    "worst_x": float(x[i]),
    "worst_ref": float(ref[i]),
    "top5_err": [float(v) for v in top.values],
    "top5_x": [float(x[int(j)]) for j in top.indices],
}))
