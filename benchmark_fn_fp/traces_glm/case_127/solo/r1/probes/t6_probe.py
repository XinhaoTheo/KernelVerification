import torch, numpy as np, json
import sys
sys.path.insert(0, "/root/cases/case_127")
import kernel

# all finite FP16 values
allbits = np.arange(65536, dtype=np.uint16)
x16 = torch.from_numpy(allbits.view(np.float16).copy())
finite = torch.isfinite(x16)
xf = x16[finite].cuda().contiguous()  # finite values, N <= 131072
print("N =", xf.numel())

out = kernel.run(xf)
ref = torch.erf(xf.to(torch.float64)).to(torch.float32)
err = (out.double() - ref.double()).abs()
maxerr = err.max().item()
worst = int(err.argmax())
viol = (err > 1e-6).sum().item()
result = {
  "metric": "max abs error vs FP64 erf reference, over ALL finite FP16 values",
  "N": int(xf.numel()),
  "max_abs_error": maxerr,
  "violations_gt_1e-6": int(viol),
  "worst_input": float(xf.cpu()[worst]),
  "worst_actual": float(out.cpu()[worst]),
  "worst_target": float(ref.cpu()[worst]),
  "tolerance": 1e-6,
  "pass": bool(viol == 0 and maxerr <= 1e-6),
}
print(json.dumps(result))
