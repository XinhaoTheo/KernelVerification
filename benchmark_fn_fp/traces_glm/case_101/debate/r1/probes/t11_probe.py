import sys, json
sys.path.insert(0, "/root/cases/case_101")
import torch, numpy as np
import kernel as K

# All FP16 subnormals: exponent field 0, mantissa 1..1023, both signs
bits = []
for m in range(1, 1024):
    for s in (0,1):
        bits.append((s<<15)|m)
bits = np.array(bits, dtype=np.uint16)
x = torch.from_numpy(bits.view(np.float16)).to("cuda")
out = K.run(x)
v = x.to(torch.float64)
r = torch.sign(v) * torch.pow(torch.abs(v), 1.0/3.0)
rel = (out.to(torch.float64) - r).abs() / r.abs()
res = {
  "metric": "max relative error vs FP64 cbrt over all 2046 signed FP16 subnormals",
  "n_subnormals": int(x.numel()),
  "max_rel_err": float(rel.max().item()),
  "n_violations": int((rel > 1e-6).sum().item()),
  "tolerance": 1e-6,
  "min_subnormal_output_sample": float(out[0].item()),
  "smallest_subnormal_ref": float(r.min().item()),
  "largest_subnormal_output_sample": float(out[-1].item()) if x[-1].item() > 0 else float(out[1022].item()),
}
print(json.dumps(res))
