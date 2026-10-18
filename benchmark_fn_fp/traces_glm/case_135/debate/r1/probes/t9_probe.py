import torch, json, sys
sys.path.insert(0, "/root/cases/case_135")
import kernel

x = torch.tensor([65504.0, -65504.0, 60000.0, -60000.0], dtype=torch.float16, device="cuda")
out = kernel.run(x)
xd = x.to(torch.float64)
ref = xd / torch.sqrt(1.0 + xd * xd)
err = (out.to(torch.float64) - ref).abs()
# also examine a range of large values
xs = torch.linspace(30000, 65504, 2001, dtype=torch.float32).to(torch.float16).to("cuda")
out2 = kernel.run(xs)
xd2 = xs.to(torch.float64)
ref2 = xd2 / torch.sqrt(1.0 + xd2 * xd2)
err2 = (out2.to(torch.float64) - ref2).abs()
print(json.dumps({
    "extrema_errs": [float(e) for e in err],
    "extrema_inputs": [float(v) for v in x],
    "large_range_max_abs_err": float(err2.max()),
    "large_range_violations_gt_1e-6": int((err2 > 1e-6).sum()),
    "large_range_worst_x": float(xs[int(err2.argmax())].item()),
    "tolerance": 1e-6
}))
