import json, math, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_128/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

bits = torch.arange(0, 65536, dtype=torch.int32)
xi = torch.tensor(bits.numpy(), dtype=torch.float16).cuda()  # every FP16 bit pattern
xi = xi[torch.isfinite(xi)]
vals = xi.cpu().numpy()
n = xi.numel()
# run in chunks under 131072
outs = []
for i in range(0, n, 100000):
    chunk = xi[i:i+100000].clone()
    outs.append(k.run(chunk).cpu())
out = torch.cat(outs).double()
ref = k.reference(xi.cpu()).double()
err = (out - ref).abs()
tol = 1e-6
worst = int(torch.argmax(err))
iab = err.argsort(descending=True)[:10]
res = {
    "n_values": int(n),
    "max_abs_err": float(err.max()),
    "tolerance": tol,
    "n_violations": int((err > tol).sum()),
    "worst_x": float(xi[worst]),
    "worst_actual": float(out[worst]),
    "worst_ref": float(ref[worst]),
    "top10": [{"x": float(xi[int(j)]), "err": float(err[int(j)]), "actual": float(out[int(j)]), "ref": float(ref[int(j)])} for j in iab],
    "finite_output": bool(torch.isfinite(out).all()),
}
print(json.dumps(res))