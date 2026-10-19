import torch, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_147/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# upper branch: all fp16 values with |x|>=3 plus extremum 65504
vals = torch.arange(0, 1<<16, dtype=torch.int32).to(torch.uint16).view(torch.float16)
vals = vals[torch.isfinite(vals)]
pos = vals[vals >= 3.0]
xs = pos.to("cuda")
N = xs.numel()
maxerr = 0.0; argx = 0.0; nviol = 0
chunk = 30000
for i in range(0, N, chunk):
    c = xs[i:i+chunk]
    o = k.run(c)
    ref = torch.atan(c.to(torch.float64))
    e = (o.to(torch.float64) - ref).abs()
    m = e.max().item()
    j = int(e.argmax())
    if m > maxerr: maxerr = m; argx = float(c[j].item())
    nviol += int((e > 1e-6).sum())
print(json.dumps({"probe": "upper branch |x|>=3 all FP16 values + extremum",
 "n": int(N), "max_abs_err": maxerr, "tolerance": 1e-6,
 "exceeds_tol": maxerr > 1e-6, "argmax_x": argx, "n_violations": nviol,
 "err_at_65504": float((k.run(torch.tensor([65504.0], device="cuda", dtype=torch.float16))[0].to(torch.float64) - torch.atan(torch.tensor(65504.0, dtype=torch.float64))).abs().item())}))
