
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_17/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(1)
res = []
for (b, s, ch) in [(3,40,64), (1,1,64), (2,63,64)]:
    x = torch.randn(b, s, device="cuda", dtype=torch.float32)
    ref = torch.cumsum(x, dim=1)
    out = m.chunked_cumsum(x, chunk=ch)
    res.append(dict(batch=b, seqlen=s, chunk=ch, n_chunks=s//ch,
                    out_nonzero_count=int((out != 0).sum()),
                    out_numel=int(out.numel()),
                    out_bitwise_all_zero=bool((out == 0).all()),
                    max_abs_err=float((out - ref).abs().max()),
                    ref_max_abs=float(ref.abs().max()),
                    ref_all_zero=bool((ref == 0).all())))
print(json.dumps({"cases": res}))
