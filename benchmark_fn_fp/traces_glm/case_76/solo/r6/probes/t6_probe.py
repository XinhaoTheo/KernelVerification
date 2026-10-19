import torch, struct, sys
sys.path.insert(0, '/root/cases/case_149')
import kernel

# All finite FP16 bit patterns, batched under the 131072 size limit
codes = [i for i in range(65536) if i not in (0x7C00, 0xFC00) and not (0x7C01 <= i <= 0x7FFF) and not (0xFC01 <= i <= 0xFFFF)]
# That includes subnormals & zeros & max finite. Build one tensor of 65504 values? Limit is 131072 so one tensor fits.
vals = torch.tensor(codes, dtype=torch.int16).view(torch.float16)
print("count:", vals.numel())
x = vals.cuda()
out = kernel.run(x)
ref = kernel.reference(x)
mismatch = (out.view(torch.int16) != ref.view(torch.int16))
n = int(mismatch.sum())
print("mismatches:", n)
if n:
    idx = mismatch.nonzero().flatten()[:20]
    for i in idx.tolist():
        xi = x[i].item(); print(hex(codes[i]), xi, out[i].item(), out[i].view(torch.int16).item() & 0xFFFF, ref[i].item(), ref[i].view(torch.int16).item() & 0xFFFF)
import json
print(json.dumps({"mismatches": n, "tested": vals.numel()}))