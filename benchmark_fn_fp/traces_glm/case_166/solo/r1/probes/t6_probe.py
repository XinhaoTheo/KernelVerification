import torch, math, json, sys
sys.path.insert(0, "/root/cases/case_166")
import kernel

bits = torch.arange(65536, dtype=torch.int32)
allvals = bits.to(torch.uint16).view(torch.float16).float()
# legal: finite and > -1 (in exact fp16 sense)
legal = torch.isfinite(allvals)
v16 = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
leg = torch.isfinite(v16) & (v16.float() > -1.0)
x = v16[leg]
print("legal count", x.numel())
outs = []
refs = []
xs = []
CH = 131072
for i in range(0, x.numel(), CH):
    chunk = x[i:i+CH].contiguous()
    out = kernel.run(chunk)
    ref = torch.log1p(chunk.to(torch.float64))
    outs.append(out.cpu())
    refs.append(ref.cpu())
    xs.append(chunk.cpu())
out = torch.cat(outs)
ref = torch.cat(refs)
xall = torch.cat(xs)
err = (out.to(torch.float64) - ref).abs()
print(json.dumps({
  "metric": "max abs error over exhaustive legal FP16 domain",
  "n": xall.numel(),
  "max_abs_err": float(err.max()),
  "violations_gt_1e-6": int((err > 1e-6).sum()),
  "worst_x": float(xall[err.argmax()]),
  "worst_out": float(out[err.argmax()]),
  "worst_ref": float(ref[err.argmax()]),
}))
# show a few worst cases
k = err.topk(5)
print("top5 err", [float(v) for v in k.values], "x", [float(xall[j]) for j in k.indices])
