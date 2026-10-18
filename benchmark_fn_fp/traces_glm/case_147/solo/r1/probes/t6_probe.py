import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

# all finite FP16 bit patterns
bits = np.arange(0, 65536, dtype=np.uint16)
allvals = torch.from_numpy(bits.view(np.float16).copy()).to(torch.float32)
finite = torch.isfinite(allvals)
vals = allvals[finite]  # 65534 finite values
print("num finite fp16:", vals.numel())

errs = []
B = 30000
ref = torch.atan(vals.to(torch.float64))
outs = []
for i in range(0, vals.numel(), B):
    chunk = vals[i:i+B].to(torch.float16).cuda()
    out = kernel.run(chunk)
    outs.append(out.cpu())
actual = torch.cat(outs).to(torch.float64)
err = (actual - ref).abs()
maxerr = err.max().item()
idx = int(err.argmax())
res = {
  "metric": "max absolute error over all finite FP16 values vs FP64 atan",
  "num_values": int(vals.numel()),
  "max_abs_err": maxerr,
  "tolerance": 1e-6,
  "violations": int((err > 1e-6).sum()),
  "worst_input": float(vals[idx]),
  "worst_actual": float(actual[idx]),
  "worst_ref": float(ref[idx]),
}
print(json.dumps(res))
# also error profile by magnitude region
for name, mask in [("|x|<0.66", vals.abs()<=0.66), ("0.66<|x|<=2.414", (vals.abs()>0.66)&(vals.abs()<=2.414213562373095)), ("|x|>2.414", vals.abs()>2.414213562373095)]:
    m = mask.cpu()
    res[name] = float(err[m].max())
print(json.dumps({k: v for k, v in res.items()}))