import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_132")
import kernel

# all finite fp16 values
vals = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
mask = np.isfinite(vals)
fp16 = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.float16)[mask].copy())
print("num finite fp16 values:", fp16.numel())

# run in batches within N<=131072
out_parts = []
B = 65536
for i in range(0, fp16.numel(), B):
    chunk = fp16[i:i+B].cuda()
    out_parts.append(kernel.run(chunk).cpu())
actual = torch.cat(out_parts)
ref = torch.sin(fp16.to(torch.float64))
err = (actual.to(torch.float64) - ref).abs()
res = {
    "metric": "max_abs_error over entire finite FP16 domain",
    "max_abs_error": float(err.max()),
    "num_violations_1e-6": int((err > 1e-6).sum()),
    "worst_input": float(fp16[err.argmax()]),
    "ref_at_worst": float(ref[err.argmax()]),
    "actual_at_worst": float(actual[err.argmax()]),
    "n": int(fp16.numel()),
}
print(json.dumps(res))