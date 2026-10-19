import torch, json, sys
sys.path.insert(0, "/root/cases/case_133")
import kernel

vals = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16).view(torch.float16).to(torch.float64)
vals = vals[torch.isfinite(vals)]
big = vals[vals.abs() >= 1024.0]
n = int(big.numel())
print("total large values:", n, file=sys.stderr)
allviol = []
maxerr = 0.0
CH = 131072
for i in range(0, n, CH):
    chunk = big[i:i+CH].to("cuda").to(torch.float16)
    out = kernel.run(chunk)
    ref = torch.sin(chunk.to(torch.float64))
    err = (out.to(torch.float64) - ref).abs()
    maxerr = max(maxerr, float(err.max().item()))
    vi = torch.nonzero(err > 1e-6).flatten()
    for j in vi.tolist():
        allviol.append({"x": float(chunk[j].item()), "err": float(err[j].item()),
                        "actual": float(out[j].item()), "ref": float(ref[j].item())})
res = {"tolerance": 1e-6, "n_tested": n, "n_violations": len(allviol),
       "max_err": maxerr, "examples": allviol[:15]}
print(json.dumps(res))
