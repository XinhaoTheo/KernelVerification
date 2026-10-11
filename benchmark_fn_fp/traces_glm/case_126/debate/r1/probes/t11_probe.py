import torch, json
import sys
sys.path.insert(0, "/root/cases/case_126")
import kernel as K

x = torch.tensor([65504.0, -65504.0, 40000.0, -40000.0, 120.0, -120.0], dtype=torch.float16).cuda()
out = K.run(x)
ref = torch.erf(x.to(torch.float64)).to(torch.float32)
err = (out.double() - ref.double()).abs()
print(json.dumps({
  "metric": "extrema_and_large_inputs",
  "inputs": x.tolist(),
  "kernel_out": out.tolist(),
  "ref": ref.tolist(),
  "abs_err": err.tolist(),
  "all_finite": bool(torch.isfinite(out).all().item()),
  "max_abs_error": err.max().item(),
  "tolerance": 1e-6,
}))