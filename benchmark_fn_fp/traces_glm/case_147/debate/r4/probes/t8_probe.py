import sys, json, torch, struct
sys.path.insert(0, "/root/cases/case_147")
import kernel

# every finite FP16 bit pattern (0x0000..0x7BFF positive, mirrored for negative)
codes = list(range(0x0000, 0x7C00))  # 0..31743 finite/positive incl +0
vals = []
for c in codes:
    vals.append(struct.unpack("<e", struct.pack("<H", c))[0])
    neg = c | 0x8000
    vals.append(struct.unpack("<e", struct.pack("<H", neg))[0])
x = torch.tensor(vals, dtype=torch.float16, device="cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
mx, idx = err.max(dim=0)
print(json.dumps({
  "n_tested": x.numel(),
  "max_abs_err": float(mx),
  "tolerance": 1e-6,
  "worst_input": float(x[idx]),
  "n_violations": int((err > 1e-6).sum()),
  "passes_contract_every_value": bool((err <= 1e-6).all()),
}))