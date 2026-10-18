import torch, importlib.util, json, sys
sys.path.insert(0, "/root/cases/case_147")
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_147/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# all positive finite fp16 values (atan odd; negatives symmetric)
bits = torch.arange(0, 1<<16, dtype=torch.int32)
allf = bits.to(torch.uint16).view(torch.float16)
allf = allf[torch.isfinite(allf)]
pos = allf[allf > 0]
N = pos.numel()
out = torch.empty(N, dtype=torch.float32, device="cuda")
# run kernel in chunks to keep N<=131072 per call
xs = pos.to("cuda")
results = []
chunk = 30000
maxerr = 0.0; argx = 0.0
for i in range(0, N, chunk):
    c = xs[i:i+chunk]
    o = k.run(c)
    ref = torch.atan(c.to(torch.float64))
    e = (o.to(torch.float64) - ref).abs()
    m = e.max().item()
    j = e.argmax().item()
    if m > maxerr:
        maxerr = m; argx = c[j].item()
print(json.dumps({"probe": "exhaustive positive finite FP16", "num_values": int(N),
 "max_abs_err": maxerr, "tolerance": 1e-6, "exceeds_tol": maxerr > 1e-6,
 "argmax_x": argx}))
