import torch, json, sys
sys.path.insert(0, "/root/cases/case_166")
import kernel

# All finite FP16 values strictly greater than -1
vals = []
for b in range(0x0000, 0x10000):
    t = torch.tensor([b], dtype=torch.uint16)
    f = t.view(torch.float16).item()
    if f == f and abs(f) != float('inf') and f > -1.0:
        vals.append(b)
x = torch.tensor(vals, dtype=torch.uint16).view(torch.float16).cuda()
ref = torch.log1p(x.to(torch.float64))
out = kernel.run(x)
err = (out.to(torch.float64) - ref).abs()
bad = err > 1e-6
res = {
    "metric": "max absolute error vs FP64 log1p over exhaustive legal FP16 domain",
    "n_legal": len(vals),
    "max_abs_err": err.max().item(),
    "n_violations": int(bad.sum().item()),
    "tolerance": 1e-6,
}
if bad.any():
    idxs = bad.nonzero().squeeze(1)[:10]
    res["examples"] = [
        {"x": x[i].item(), "actual": out[i].item(), "ref": ref[i].item(), "err": err[i].item()} for i in idxs
    ]
print(json.dumps(res))
