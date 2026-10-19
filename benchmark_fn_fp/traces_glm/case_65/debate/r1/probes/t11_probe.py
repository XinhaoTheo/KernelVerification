import sys, json, torch
sys.path.insert(0, "/root/cases/case_138")
import kernel as K

# contiguous view with nonzero storage offset, N=4093 (not multiple of 256)
N, KOFF = 4093, 7
base = (2.0*torch.randn(5000, generator=torch.Generator().manual_seed(1))).to(torch.float16).cuda()
x = base[KOFF:KOFF+N]
assert x.is_contiguous() and x.storage_offset() == KOFF
before = x.clone()
out = K.run(x)
r64 = x.to(torch.float64); sq = r64*r64; rho = torch.sqrt(1.0+sq)-1.0
err = (out.to(torch.float64)-rho).abs(); tol = 1e-8 + 1e-5*rho.abs()
input_preserved = bool(torch.equal(before, x))
# also test N=1 and N=131072 with offset
base2 = (2.0*torch.randn(140000, generator=torch.Generator().manual_seed(2))).to(torch.float16).cuda()
x2 = base2[3:3+131072]
out2 = K.run(x2)
rho2 = torch.sqrt(1.0 + x2.to(torch.float64)**2) - 1.0
err2 = (out2.to(torch.float64)-rho2).abs(); tol2 = 1e-8 + 1e-5*rho2.abs()
x1 = base2[5:6]
out1 = K.run(x1)
rho1 = torch.sqrt(1.0 + x1.to(torch.float64)**2) - 1.0
err1 = (out1.to(torch.float64)-rho1).abs(); tol1 = 1e-8 + 1e-5*rho1.abs()
print(json.dumps({"offset": KOFF, "n": N, "violations_n4093": int((err>tol).sum()), "input_preserved": input_preserved,
  "n_max": 131072, "violations_n131072": int((err2>tol2).sum()), "n1": 1, "violations_n1": int((err1>tol1).sum()),
  "tail_last5_err": err[-5:].tolist(), "head_first5_err": err[:5].tolist()}))
