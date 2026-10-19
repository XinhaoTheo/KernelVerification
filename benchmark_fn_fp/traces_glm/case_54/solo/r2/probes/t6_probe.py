import torch, json, sys
sys.path.insert(0, "/root/cases/case_127")
import kernel

# All finite FP16 bit patterns
bits = torch.arange(65536, dtype=torch.int64)
hf = bits.to(torch.uint16).view(torch.int16).to(torch.float16).view(-1)
finite = torch.isfinite(hf)
x_all = hf[finite].cuda()

# run in chunks to respect N<=131072 validation
ref = torch.erf(x_all.to(torch.float64))
worst = 0.0; worst_x = 0.0; errs = []
for i in range(0, x_all.numel(), 100000):
    out = kernel.run(x_all[i:i+100000])
    r = ref[i:i+100000].to(torch.float32).to(torch.float64)
    e = (out.to(torch.float64) - r).abs()
    m = e.max().item()
    if m > worst:
        worst = m; idx = i + e.argmax().item(); worst_x = x_all[idx].item()
    errs.append(e.max().item())

over = 0
# count violations precisely
for i in range(0, x_all.numel(), 100000):
    out = kernel.run(x_all[i:i+100000])
    r = ref[i:i+100000].to(torch.float32).to(torch.float64)
    over += int(((out.to(torch.float64) - r).abs() > 1e-6).sum().item())

print(json.dumps({"metric":"max abs error vs FP64->FP32 erf over all finite FP16 values",
 "num_tested": x_all.numel(), "max_abs_err": worst, "worst_x": worst_x,
 "num_violations_over_1e-6": over, "tolerance": 1e-6}))
