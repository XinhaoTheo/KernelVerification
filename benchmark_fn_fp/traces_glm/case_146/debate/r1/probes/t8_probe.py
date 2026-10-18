
import torch, sys, json
sys.path.insert(0, "/root/cases/case_146")
from kernel import run

# All finite FP16 bit patterns (excluding NaN/Inf)
bits = torch.arange(65536, dtype=torch.int32)
fp16 = bits.to(torch.uint16).view(torch.float16).view(-1)
finite = torch.isfinite(fp16)
vals = fp16[finite].cuda()  # all 65536 finite values incl. both zeros
print("num finite values:", vals.numel(), file=sys.stderr)

out = run(vals)  # fp32
ref = torch.atan(vals.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
max_err = err.max().item()
n_fail = int((err > 1e-6).sum().item())
idx = int(err.argmax().item())
result = {
    "metric": "max abs error vs FP64 atan over ALL finite FP16 values",
    "tolerance": 1e-6,
    "n_tested": vals.numel(),
    "max_abs_err": max_err,
    "n_exceeding_tol": n_fail,
    "worst_input": float(vals[idx].item()),
    "worst_input_abs": abs(float(vals[idx].item())),
    "worst_kernel_out": float(out[idx].item()),
    "worst_ref": float(ref[idx].item()),
}
print(json.dumps(result))
