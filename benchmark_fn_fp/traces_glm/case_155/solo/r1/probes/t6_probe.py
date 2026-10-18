
import torch, json, sys
sys.path.insert(0, "/root/cases/case_155")
import kernel as K

dev = "cuda"
cases = {}

# Case A: extreme magnitudes with true real ~0 (a*c + b*d cancellation) but imag large
num = torch.tensor([[65504.0, 65504.0],[1000.0, 3000.0]], dtype=torch.float16)
den = torch.tensor([[3.0, -3.0],[7.0, 21.0]], dtype=torch.float16)
num, den = num.to(dev), den.to(dev)
out = K.run(num, den)
ref = K.reference(num.cpu(), den.cpu()).to(dev)
err = (out - ref).abs()
tol = 1e-7 + 5e-7 * ref.abs()
cases["A"] = {"err": err.max().item(), "tol_min": tol.min().item(), "worst_ratio": (err/tol).max().item()}

# Case B: ratio rounding: real_numerator = fma(b, ratio, a) with b*ratio ~ a -> real ~ 0
num = torch.tensor([[65504.0, -65504.0]], dtype=torch.float16).to(dev)
den = torch.tensor([[65504.0, 65504.0]], dtype=torch.float16).to(dev)
out = K.run(num, den)
ref = K.reference(num.cpu(), den.cpu()).to(dev)
err = (out - ref).abs(); tol = 1e-7 + 5e-7*ref.abs()
cases["B"] = {"err": err.max().item(), "tol_min": tol.min().item(), "ratio": (err/tol).max().item(), "out": out.tolist(), "ref": ref.tolist()}

# Case C: ratio near 1, cancellation: a*c+b*d small vs terms huge, denominator small components
num = torch.tensor([[60000.0, 60000.0]], dtype=torch.float16).to(dev)
den = torch.tensor([[1.0, -1.0]], dtype=torch.float16).to(dev)
out = K.run(num, den); ref = K.reference(num.cpu(), den.cpu()).to(dev)
err = (out - ref).abs(); tol = 1e-7 + 5e-7*ref.abs()
cases["C"] = {"err": err.max().item(), "ratio": (err/tol).max().item()}

# Case D: random sweep of large-magnitude fp16 values, exhaustive-ish grid on a,b with ratio rounding
import itertools
vals = [65504.0, 1000.0, -65504.0, -1000.0, 60000.0, -60000.0]
worst = 0.0; worst_info=None
N_num=[]; N_den=[]; Ns=[]
for c_ in [65504.0, 100.0, 1.0, 60000.0, -65504.0, -100.0]:
    for d_ in [65504.0, 100.0, 1.0, 60000.0, -65504.0, -100.0]:
        if c_==0 and d_==0: continue
        for a_ in vals:
            for b_ in vals:
                N_num.append([a_,b_]); N_den.append([c_,d_]); Ns.append(True)
num = torch.tensor(N_num, dtype=torch.float16).to(dev)
den = torch.tensor(N_den, dtype=torch.float16).to(dev)
out = K.run(num, den)
ref = K.reference(num.cpu(), den.cpu()).to(dev)
err = (out - ref).abs(); tol = 1e-7 + 5e-7*ref.abs()
ratio = (err/tol)
i = (ratio.max(dim=1).values.argmax()).item()
cases["D"] = {"max_ratio": ratio.max().item(), "worst_row": {"num": N_num[i], "den": N_den[i], "out": out[i].tolist(), "ref": ref[i].tolist(), "err": err[i].tolist()}}

# Case E: subnormals / tiny denominator, ratio-based
num = torch.tensor([[5.96e-8, -5.96e-8],[1.0,2.0]], dtype=torch.float16).to(dev)  # subnormal
den = torch.tensor([[5.96e-8, 5.96e-8],[3.0,4.0]], dtype=torch.float16).to(dev)
out = K.run(num, den); ref = K.reference(num.cpu(), den.cpu()).to(dev)
err = (out - ref).abs(); tol = 1e-7 + 5e-7*ref.abs()
cases["E"] = {"max_ratio": (err/tol).max().item()}

print(json.dumps(cases, indent=2))
