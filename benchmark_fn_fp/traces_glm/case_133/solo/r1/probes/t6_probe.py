
import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_133")
import kernel

# all finite fp16 bit patterns
bits = np.arange(0, 65536, dtype=np.uint16)
vals = np.frombuffer(bits.tobytes(), dtype=np.float16)
finite = np.isfinite(vals)
x = torch.from_numpy(vals[finite]).cuda()
out = kernel.run(x)
ref = kernel.reference(x)
err = (out.to(torch.float64) - ref).abs()
bad = err > 1e-6
res = {
  "n": int(x.numel()),
  "max_abs_err": float(err.max()),
  "n_violations": int(bad.sum()),
  "violation_fraction": float(bad.float().mean()),
  "worst_x": float(x[err.argmax()].item()),
  "worst_out": float(out[err.argmax()].item()),
  "worst_ref": float(ref[err.argmax()].item()),
}
print(json.dumps(res))
