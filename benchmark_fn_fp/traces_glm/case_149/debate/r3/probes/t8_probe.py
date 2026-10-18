
import sys, os, json
sys.path.insert(0, "/root/cases/case_149")
import torch, kernel

# all finite FP16 bit patterns (exclude NaN/Inf exponent=0x1F except zero subnormals are fine)
patterns = [i for i in range(65536) if not (i >> 10) == 0x1F]
x = torch.tensor(patterns, dtype=torch.int32).view(torch.float16).to("cuda")
assert x.numel() <= 131072 and torch.isfinite(x).all()

out = kernel.run(x)
ref = kernel.reference(x)
ok = out.view(torch.int16) == ref.view(torch.int16)
mismatches = (~ok).nonzero().flatten()
n_mismatch = int(mismatches.numel())
fail = []
for idx in mismatches[:10].tolist():
    xv = x[idx].item()
    fail.append({"x": xv, "x_bits": hex(x[idx].view(torch.int16).item()),
                 "got_bits": hex(out[idx].view(torch.int16).item()),
                 "got": out[idx].item(),
                 "want_bits": hex(ref[idx].view(torch.int16).item()),
                 "want": ref[idx].item()})
print(json.dumps({"n_tested": x.numel(), "n_mismatch": n_mismatch,
                  "first_failures": fail}))
