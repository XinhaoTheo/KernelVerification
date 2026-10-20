import sys, os, json, struct
sys.path.insert(0, "/root/cases/case_101")
import torch, numpy as np

def kernel_run(x):
    # direct kernel launch on candidate tensor (bypass run() validation limits not needed; N fits)
    import kernel as K
    return K.run(x)

def ref(x):
    v = x.to(torch.float64)
    return torch.sign(v) * torch.pow(torch.abs(v), 1.0/3.0)

# all finite FP16 bit patterns: skip exponent 0x1F (Inf/NaN) => 15*1024*2 = 30720 values
bits = []
for e in range(31):
    for m in range(1024):
        for s in (0,1):
            if e == 31 and m > 0: continue
            if e == 31 and m == 0: continue  # Inf
            bits.append((s<<15)|(e<<10)|m)
bits = np.array(bits, dtype=np.uint16)
xf = torch.from_numpy(bits.view(np.float16)).to("cuda")

out = kernel_run(xf)
r = ref(xf)
nz = xf != 0
rel = (out[nz].to(torch.float64) - r[nz]).abs() / r[nz].abs()
maxrel = rel.max().item()
viol = (rel > 1e-6).sum().item()
argmax = rel.argmax().item()
worst_x = xf[nz][argmax].item()
res = {
  "metric": "max relative error vs FP64 cbrt over all finite FP16 values (30864 positive+negative, zeros excluded)",
  "n_values": int(xf.numel()),
  "max_rel_err": maxrel,
  "tolerance": 1e-6,
  "n_violations": int(viol),
  "worst_input": float(worst_x),
  "kernel_output_at_worst": float(out[nz][argmax].item()),
  "ref_at_worst": float(r[nz][argmax].item()),
}
print(json.dumps(res))
