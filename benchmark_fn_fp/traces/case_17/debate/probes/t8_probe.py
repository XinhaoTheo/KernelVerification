
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_17/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = []
for (b, s, ch) in [(4,100,64), (2,65,64), (2,200,64), (2,128,64)]:
    x = torch.randn(b, s, device="cuda", dtype=torch.float32)
    ref = torch.cumsum(x, dim=1)
    out = m.chunked_cumsum(x, chunk=ch)
    nc = s // ch
    cut = nc * ch
    head_err = float((out[:, :cut] - ref[:, :cut]).abs().max()) if cut > 0 else None
    if cut < s:
        tail_out = out[:, cut:]
        tail_ref = ref[:, cut:]
        tail_err = float((tail_out - tail_ref).abs().max())
        tail_all_zero = bool((tail_out == 0).all())
        tail_ref_absmin = float(tail_ref.abs().min())
        n_tail = tail_out.numel()
    else:
        tail_err = None; tail_all_zero = None; tail_ref_absmin = None; n_tail = 0
    res.append(dict(batch=b, seqlen=s, chunk=ch, n_chunks=nc, cut=cut,
                    head_max_abs_err=head_err, tail_max_abs_err=tail_err,
                    tail_bitwise_all_zero=tail_all_zero,
                    tail_ref_min_abs=tail_ref_absmin, n_tail_elems=n_tail,
                    global_allclose=bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-4))))
print(json.dumps({"cases": res}))
