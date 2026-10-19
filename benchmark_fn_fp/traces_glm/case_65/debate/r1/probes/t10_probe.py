import sys, json, torch
sys.path.insert(0, "/root/cases/case_138")
import kernel as K

# all FP16 subnormal bit patterns + signed zeros + crossover region values
vals = []
for b in range(1, 1024):  # subnormal magnitudes
    v = torch.tensor(b, dtype=torch.uint16).view(torch.float16).item()
    vals += [v, -v]
vals += [0.0, -0.0]
for b in [1024, 1536, 1792, 2048, 2560, 3072, 4096, 5120, 6144, 7168]:  # values ~6e-5..1.4e-4 region
    v = torch.tensor(b, dtype=torch.uint16).view(torch.float16).item()
    vals += [v, -v]
xs = torch.tensor(vals, dtype=torch.float16, device="cuda")
out = K.run(xs)
r64 = xs.to(torch.float64); sq = r64*r64; rho = torch.sqrt(1.0+sq)-1.0
err = (out.to(torch.float64)-rho).abs()
tol = 1e-8 + 1e-5*rho.abs()
viol = err > tol
idx = viol.nonzero().flatten().tolist()
print(json.dumps({"n": xs.numel(), "num_violations": int(viol.sum()), "violation_indices_first10": idx[:10],
  "max_err": err.max().item(), "max_ratio_err_over_tol": (err/tol).max().item(),
  "zero_out_plus": out[vals.index(0.0)].item(), "zero_out_minus": out[vals.index(-0.0)].item()}))
