import sys, json, torch
sys.path.insert(0, "/root/cases/case_147")
import kernel

# all finite FP16 values
bits = torch.arange(65536, dtype=torch.int32)
vals = bits.to(torch.uint16).view(torch.float16) if hasattr(torch,'uint16') else None
# build via numpy-free approach
import struct
xs = []
for b in range(65536):
    xs.append(torch.tensor([0], dtype=torch.float16))
# simpler: use torch view trick
u = torch.arange(65536, dtype=torch.int64)
buf = u.to(torch.int16).cpu().numpy()
t = torch.from_numpy(buf.copy()).view(torch.float16)
mask = torch.isfinite(t)
t = t[mask].to("cuda")
out = kernel.run(t)
ref = torch.atan(t.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
mx = err.max().item()
idx = err.argmax().item()
print(json.dumps({
  "metric": "max abs error over all finite FP16 values (exhaustive)",
  "max_abs_err": mx,
  "tolerance": 1e-6,
  "n_tested": t.numel(),
  "worst_x": t[idx].item(),
  "worst_out": out[idx].item(),
  "worst_ref": ref[idx].item(),
  "n_violations": int((err > 1e-6).sum().item()),
  "all_finite_out": bool(torch.isfinite(out).all().item()),
  "shape_dtype": [list(out.shape), str(out.dtype)],
  "reason": "contract requires abs(actual-atan(x)) <= 1e-6 for every finite FP16 value"
}))