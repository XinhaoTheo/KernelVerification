import torch, sys
sys.path.insert(0, '/root/cases/case_101')
import kernel

bits = torch.arange(65536, dtype=torch.int32)
x16 = bits.to(torch.uint16).view(torch.float16).cpu()
finite = torch.isfinite(x16)
xf = x16[finite].to('cuda')  # keep fp16

out = kernel.run(xf)

xd = xf.cpu().to(torch.float64)
ref = torch.sign(xd) * torch.pow(torch.abs(xd), 1.0/3.0)
actual = out.cpu().to(torch.float64)

nonzero = xd != 0
rel = (actual[nonzero] - ref[nonzero]).abs() / ref[nonzero].abs()
maxrel = rel.max().item()
worst_bits = None
worst_x = None
if maxrel > 1e-6:
    idx = int(rel.argmax())
    wi = int(torch.where(nonzero)[0][idx])
    worst_bits = int(bits[finite][wi])
    worst_x = float(x16[finite][wi])
zero_ok = bool((actual[~nonzero] == 0).all().item())
print({'num_tested': int(xf.numel()), 'max_rel_err': maxrel, 'tol': 1e-6,
       'passes': maxrel <= 1e-6, 'zero_ok': zero_ok, 'worst_bits': worst_bits, 'worst_x': worst_x})
