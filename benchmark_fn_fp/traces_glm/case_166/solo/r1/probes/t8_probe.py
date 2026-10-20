import torch, json, sys
sys.path.insert(0, "/root/cases/case_166")
import kernel

v16 = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
leg = torch.isfinite(v16) & (v16.float() > -1.0)
x = v16[leg].contiguous()
print("legal count", x.numel())
outs, refs, xs = [], [], []
CH = 131072
for i in range(0, x.numel(), CH):
    chunk = x[i:i+CH].to("cuda").contiguous()
    out = kernel.run(chunk)
    ref = torch.log1p(chunk.cpu().to(torch.float64))
    outs.append(out.cpu()); refs.append(ref); xs.append(chunk.cpu())
out = torch.cat(outs); ref = torch.cat(refs); xall = torch.cat(xs)
err = (out.to(torch.float64) - ref).abs()
k = err.topk(5)
print(json.dumps({
  "metric": "max abs error over exhaustive legal FP16 domain (> -1), vs FP64 log1p",
  "n": xall.numel(),
  "max_abs_err": float(err.max()),
  "violations_gt_1e-6": int((err > 1e-6).sum()),
  "worst_x": float(xall[err.argmax()]),
  "worst_out": float(out[err.argmax()]),
  "worst_ref": float(ref[err.argmax()]),
  "top5_err": [float(v) for v in k.values],
  "top5_x": [float(xall[j]) for j in k.indices],
}))