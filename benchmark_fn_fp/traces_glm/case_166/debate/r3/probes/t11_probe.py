
import sys, json, torch
sys.path.insert(0, "/root/cases/case_166")
from kernel import run

# values just above -1 plus subnormal FP16 x, vs FP64 log1p
bits = torch.arange(65536, dtype=torch.int32)
x_all = bits.to(torch.uint16).view(torch.float16)
neg_near = x_all[(x_all > -1.0) & (x_all < -0.9)]
sub = x_all[(x_all.abs() > 0) & (x_all.abs() < 6.104e-5)]  # subnormal magnitudes
tiny_norm = x_all[(x_all.abs() >= 6.104e-5) & (x_all.abs() < 1e-3)]
vals = torch.cat([neg_near, sub, tiny_norm, torch.tensor([0.0, -0.0, -0.99951171875])]).to(torch.float16)
x = vals.to("cuda")
out = run(x)
ref = torch.log1p(vals.cpu().to(torch.float64))
err = (out.cpu().to(torch.float64) - ref).abs()
maxerr = float(err.max())
i = int(err.argmax())
submask = torch.cat([torch.zeros(len(neg_near), dtype=torch.bool), torch.ones(len(sub)+len(tiny_norm)+2, dtype=torch.bool)])
print(json.dumps({
    "metric": "max abs error for x near -1 and subnormal/small FP16 x vs FP64 log1p",
    "n_tested": int(vals.numel()),
    "n_near_minus1": int(len(neg_near)),
    "n_subnormal_or_tiny": int(len(sub)+len(tiny_norm)),
    "max_abs_err": maxerr,
    "tolerance": 1e-6,
    "exceeds_1e-6": bool(maxerr > 1e-6),
    "worst_x": float(vals[i]),
    "worst_ref": float(ref[i]),
}))
