import torch, struct, sys, os
sys.path.insert(0, "/root/cases/case_149")
import kernel

bits = torch.arange(0, 65536, dtype=torch.int32)
x = bits.to(torch.uint16).view(torch.float16)
finite = torch.isfinite(x)
xf = x[finite]  # all finite FP16 encodings

# reference: FP64 sigmoid then round once to binary16
vals = xf.cpu().to(torch.float64)
sig = torch.sigmoid(vals)
ref = torch.tensor([struct.unpack("<e", struct.pack("<e", v))[0] for v in sig.tolist()], dtype=torch.float16)

out = kernel.run(xf.cuda())
out = out.cpu()
mismatch = (out != ref)
n = int(mismatch.sum())
print("total_finite:", int(finite.numel()))
print("mismatches:", n)
if n:
    idx = mismatch.nonzero().flatten()[:20]
    for i in idx.tolist():
        xi = xf[i].item(); o = out[i].item(); r = ref[i].item()
        s64 = torch.sigmoid(torch.tensor(float(xi), dtype=torch.float64)).item()
        print(f"x={xi!r} bits={hex(xf[i].view(torch.int16).item())} kernel={o!r} bits={hex(out[i].view(torch.int16).item())} ref={r!r} bits={hex(ref[i].view(torch.int16).item())} fp64={s64!r}")
import json
print(json.dumps({"metric":"exact FP16 bit equality vs FP64-round-once reference","mismatches":n,"domain_size":int(finite.numel())}))
