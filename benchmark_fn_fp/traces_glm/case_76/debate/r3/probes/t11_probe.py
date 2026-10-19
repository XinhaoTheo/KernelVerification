import sys, json
sys.path.insert(0, "/root/cases/case_149")
import torch, kernel

# all finite FP16 bit patterns via uint16 -> float16 reinterpretation (NaN/Inf excluded)
import numpy as np
bits = np.array([i for i in range(65536) if (i >> 10) != 0x1F], dtype=np.uint16)
x = torch.from_numpy(bits).view(torch.float16).to("cuda")
assert x.numel() <= 131072 and torch.isfinite(x).all()

out = kernel.run(x)
ref = kernel.reference(x)
ok = out.view(torch.int16) == ref.view(torch.int16)
mismatches = (~ok).nonzero().flatten()
n_mismatch = int(mismatches.numel())
fail = []
for idx in mismatches[:10].tolist():
    fail.append({"x": x[idx].item(),
                 "x_bits": hex(int(x[idx].view(torch.int16).item()) & 0xFFFF),
                 "got_bits": hex(int(out[idx].view(torch.int16).item()) & 0xFFFF),
                 "got": out[idx].item(),
                 "want_bits": hex(int(ref[idx].view(torch.int16).item()) & 0xFFFF),
                 "want": ref[idx].item()})
print(json.dumps({"n_tested": int(x.numel()), "n_mismatch": n_mismatch,
                  "first_failures": fail}))
