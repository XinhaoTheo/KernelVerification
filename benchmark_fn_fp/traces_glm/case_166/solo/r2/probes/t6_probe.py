import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_166")
import kernel

# all legal fp16 values: every uint16 code, decode, keep finite and > -1
bits = np.arange(0, 65536, dtype=np.uint16)
allv = torch.from_numpy(bits.view(np.float16))
legal = allv[torch.isfinite(allv) & (allv > -1)]
x = legal.clone().cuda()
print("legal count:", x.numel(), "min:", x.min().item(), "max:", x.max().item())

out = kernel.run(x)
ref = torch.log1p(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
maxerr = err.max().item()
bad = (err > 1e-6).sum().item()
i = err.argmax().item()
res = {
  "metric": "max_abs_error_over_full_legal_fp16_domain",
  "num_legal_inputs": int(x.numel()),
  "max_abs_err": maxerr,
  "num_violations_gt_1e-6": int(bad),
  "worst_input": float(x[i]),
  "worst_kernel_out": float(out[i]),
  "worst_ref": float(ref[i]),
  "tolerance": 1e-6,
  "dtype_out": str(out.dtype),
}
print(json.dumps(res))
