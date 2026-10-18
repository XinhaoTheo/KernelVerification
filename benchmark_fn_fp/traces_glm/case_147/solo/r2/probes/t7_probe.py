import sys, json, torch
sys.path.insert(0, "/root/cases/case_147")
import kernel as K

bits = torch.arange(65536, dtype=torch.int32)
bits[(bits == 0x7C00) | (bits == 0xFC00) | ((bits & 0x7F00) == 0x7F00)] = -1  # NaN/Inf excluded
bits = bits[bits >= 0].to(torch.int16)
x = bits.view(torch.float16).cuda()
N = x.numel()
errs = []
ref = []
for chunk_start in range(0, N, 65536):
    c = x[chunk_start:chunk_start+65536].contiguous()
    out = K.run(c)
    r = torch.atan(c.to(torch.float64))
    errs.append((out.to(torch.float64) - r).abs())
err = torch.cat(errs)
print(json.dumps({"N": N, "max_abs_err": float(err.max()),
  "n_violations_1e-6": int((err > 1e-6).sum()),
  "worst_bits": int(bits[err.argmax()].item()) if False else None,
  "worst_x": float(x[err.argmax()].item())}))
