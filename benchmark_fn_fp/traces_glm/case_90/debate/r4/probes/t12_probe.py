import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
res = {}
for N in (64,128,256):
    g = torch.Generator(device="cpu").manual_seed(99+N)
    # even/odd-asymmetric random input: distinct even and odd content
    x = torch.randn((4,N), generator=g).to(torch.float16)
    # make Nyquist large: add alternating component
    alt = ((torch.arange(N) % 2) * 2 - 1).float() * 5000.0
    x = (x.float() + alt).to(torch.float16).to(dev)
    act = k.run(x)
    ref = torch.fft.rfft(x.to(torch.float64), dim=1)
    nyq_act = act[:, N//2, :]
    nyq_ref = torch.stack((ref.real[:, N//2], ref.imag[:, N//2]), dim=-1)
    err = (nyq_act.to(torch.float64)-nyq_ref).abs()
    allowed = 1e-4 + 1e-6*nyq_ref.abs()
    res[N] = {"max_nyquist_err": float(err.max()), "tol": float(allowed.max()),
              "imag_exactly_zero": bool((act[:, N//2, 1] == 0.0).all()),
              "pass": bool((err <= allowed).all())}
print(json.dumps({"metric":"Nyquist bin k=N/2 abs error vs tolerance; imag part zero check","result":res}))