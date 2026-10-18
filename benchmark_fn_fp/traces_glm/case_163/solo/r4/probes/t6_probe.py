
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_163/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

results = {}
worst = 0.0
fail = False
cases = []
for n in (64,128,256):
    # random
    g = torch.Generator().manual_seed(n)
    for name, mk in [
        ("randn", lambda: torch.randn((8,n), generator=g)),
        ("extremes", lambda: torch.where(torch.rand((8,n), generator=g)<0.5, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)),
        ("subnormal", lambda: (torch.rand((8,n), generator=g)*1e-7).to(torch.float16)),
        ("mixed", lambda: (torch.randn((8,n), generator=g)*1000).to(torch.float16)),
        ("impulse", lambda: torch.eye(8, n)[..., :n].clone()*0 + torch.zeros((8,n)).index_fill_(1, torch.tensor([0]), 65504.0)),
        ("constant", lambda: torch.full((8,n), 65504.0)),
    ]:
        v = mk().to(torch.float16).to("cuda")
        out = k.run(v)
        ref = torch.fft.rfft(v.to(torch.float64), dim=1, norm="backward")
        ref = torch.stack((ref.real, ref.imag), dim=-1)
        diff = (out.to(torch.float64) - ref).abs()
        tol = 1e-4 + 1e-6*ref.abs()
        bad = (diff > tol).sum().item()
        md = diff.max().item()
        worst = max(worst, md / (tol.max().item() if tol.max()>0 else 1))
        err_idx = (diff/tol.clamp(min=1e-300)).max()
        ok = bool(torch.isfinite(out).all()) and bad==0
        cases.append({"n":n,"case":name,"num_viol":bad,"max_abs_err":diff.max().item(),"max_ratio":float(err_idx)})
        if bad>0 or not ok: fail=True
print(json.dumps({"cases":cases,"any_fail":fail,"worst_ratio":worst}))
