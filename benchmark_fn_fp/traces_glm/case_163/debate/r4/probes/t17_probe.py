import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
N = 256
j = torch.arange(N)
rows = []
# all finite extreme-magnitude patterns, each with a near-zero-cancellation bin
rows.append(65504.0 * ((j % 2) * 2 - 1).float())          # alternating +/-65504: Nyquist=65504*256, all other bins cancel to ~0
rows.append(torch.full((N,), 65504.0))                      # constant: DC=65504*256, others ~0
r = 65504.0*torch.cos(2*3.141592653589793*1*j/N) + 60000.0*torch.cos(2*3.141592653589793*3*j/N)
rows.append(r)                                              # coherent pair, most bins cancel to ~0
rows.append(32768.0*torch.cos(2*3.141592653589793*10*j/N + 0.7) - 32768.0*torch.cos(2*3.141592653589793*10*j/N))  # engineered near-total cancellation at f=10
x = torch.stack(rows).to(torch.float16)
assert torch.isfinite(x).all()
x = x.to(dev)
act = k.run(x)
ref = torch.fft.rfft(x.to(torch.float64), dim=1)
refc = torch.stack((ref.real, ref.imag), dim=-1)
err = (act.to(torch.float64)-refc).abs()
allowed = 1e-4 + 1e-6*refc.abs()
ratio = err/allowed
# report separately cancellation bins (|ref| < 1e6) and overall
small = refc.abs() < 1e6
out = {"pass_all": bool((err <= allowed).all()), "nonfinite": bool((~torch.isfinite(act)).any()),
       "worst_ratio": float(ratio.max()), "max_abs_err": float(err.max()),
       "worst_ratio_cancellation_bins": float(ratio[small].max()) if small.any() else None,
       "max_abs_err_cancellation_bins": float(err[small].max()) if small.any() else None,
       "min_abs_ref": float(refc.abs().min())}
print(json.dumps({"metric":"abs err vs 1e-4+1e-6|ref| on finite extreme-magnitude FP16 inputs N=256, cancellation bins isolated", **out}))