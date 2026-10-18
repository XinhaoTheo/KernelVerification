import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
N = 256
g = torch.Generator(device="cpu").manual_seed(7)
# coherent extreme-magnitude rows: +/-65504 sinusoid at frequency f1 combined with another
# so most bins cancel exactly, plus near-zero-cancellation via two adjacent frequencies
j = torch.arange(N)
rows = []
for (f1, f2, a, b) in [(1,2,65504.,65504.),(3,3,65504.,-65504.),(5,254,60000.,60000.),(64,32,65504.,65000.)]:
    r = a*torch.cos(2*3.141592653589793*f1*j/N) + b*torch.cos(2*3.141592653589793*f2*j/N)
    rows.append(r)
x = torch.stack(rows).to(torch.float16).to(dev)
act = k.run(x)
ref = torch.fft.rfft(x.to(torch.float64), dim=1)
refc = torch.stack((ref.real, ref.imag), dim=-1)
err = (act.to(torch.float64)-refc).abs()
allowed = 1e-4 + 1e-6*refc.abs()
ratio = err/allowed
worst = ratio.max()
# smallest |ref| magnitude present (near-zero cancellation bins)
minabs = refc.abs().min()
pass_all = bool((err <= allowed).all()) and bool(torch.isfinite(act).all())
print(json.dumps({"metric":"abs err vs 1e-4+1e-6|ref| on coherent extreme FP16 inputs N=256",
 "pass":pass_all, "worst_ratio":float(worst), "max_abs_err":float(err.max()),
 "min_abs_ref":float(minabs), "nonfinite": bool((~torch.isfinite(act)).any())}))