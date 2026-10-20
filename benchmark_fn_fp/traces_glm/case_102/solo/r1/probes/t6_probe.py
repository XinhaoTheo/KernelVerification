import sys, json, torch
sys.path.insert(0, "/root/cases/case_102")
import kernel

# all finite fp16 bit patterns
bits = torch.arange(65536, dtype=torch.int32)
bf = torch.tensor(bits.numpy().astype("uint16").view("<u2"), dtype=torch.uint16)
hf = bf.view(torch.float16)
fin = torch.isfinite(hf)
x = hf[fin].to("cuda")  # signed zeros included
n = x.numel()

out = kernel.run(x)

xd = x.to(torch.float64)
ref = torch.sign(xd) * torch.pow(torch.abs(xd), 1.0/3.0)

mask = xd != 0
rel = (out.to(torch.float64)[mask] - ref[mask]).abs() / ref[mask].abs()
max_rel = rel.max().item()
# zero handling: absolute tolerance zero
zeros = x == 0
zero_ok = bool((out[zeros] == 0).all().item())
print(json.dumps({
  "metric": "max relative error over all finite fp16 values; zero exactness",
  "n_values": int(n),
  "n_nonzero": int(mask.sum().item()),
  "n_zero": int(zeros.sum().item()),
  "max_relative_error": max_rel,
  "tolerance": 1e-6,
  "zero_inputs_exact_zero": zero_ok,
  "violations_count": int((rel > 1e-6).sum().item()),
  "worst_input": float(x[mask][rel.argmax()].item()),
  "worst_actual": float(out[mask][rel.argmax()].item()),
  "worst_target": float(ref[mask][rel.argmax()].item()),
  "subnormal_inputs": bool(((x != 0) & (x.abs() < 6.104e-05)).any().item()),
}))
