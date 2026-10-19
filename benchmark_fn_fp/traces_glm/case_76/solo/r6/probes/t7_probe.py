import torch, sys, json
sys.path.insert(0, '/root/cases/case_149')
import kernel

codes = [i for i in range(65536) if not (0x7C00 <= i <= 0x7FFF or 0xFC00 <= i <= 0xFFFF)]
vals = torch.tensor(codes, dtype=torch.int32).to(torch.int16).view(torch.float16)
print("count:", vals.numel(), "finite:", bool(torch.isfinite(vals).all()))
x = vals.cuda()
out = kernel.run(x)
ref = kernel.reference(x)
mismatch = (out.view(torch.int16) != ref.view(torch.int16))
n = int(mismatch.sum())
print("mismatches:", n)
details = []
if n:
    idx = mismatch.nonzero().flatten()[:20]
    for i in idx.tolist():
        o = int(out[i].view(torch.int16).item()) & 0xFFFF
        r = int(ref[i].view(torch.int16).item()) & 0xFFFF
        details.append({"code": hex(codes[i]), "x": x[i].item(), "out_bits": hex(o), "ref_bits": hex(r)})
        print(details[-1])
print(json.dumps({"mismatches": n, "tested": vals.numel(), "details": details}))