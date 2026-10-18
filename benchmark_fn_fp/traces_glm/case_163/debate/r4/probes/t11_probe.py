import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
tol_a, tol_r = 1e-4, 1e-6
out = {}
for N in (64,128,256):
    g = torch.Generator(device="cpu").manual_seed(1234+N)
    x = torch.randn((8,N), generator=g).to(torch.float16).to(dev)
    act = k.run(x)
    ref = torch.fft.rfft(x.to(torch.float64), dim=1)
    refc = torch.stack((ref.real, ref.imag), dim=-1)
    err = (act.to(torch.float64)-refc).abs()
    allowed = tol_a + tol_r*refc.abs()
    viol = (err > allowed)
    # worst violating bin
    if viol.any():
        idx = (err/allowed).argmax()
        out[N] = "FAIL worst ratio %.3g" % float((err/allowed).max())
    else:
        out[N] = "PASS worst ratio %.3g" % float((err/allowed).max())
print(json.dumps({"metric":"per-component abs err vs FP64 rfft, ratio err/(1e-4+1e-6|ref|)","widths":out,"nonfinite": any(not torch.isfinite(a).all() for a in [k.run(torch.randn((2,N)).to(torch.float16).to(dev)) for N in (64,)])}))