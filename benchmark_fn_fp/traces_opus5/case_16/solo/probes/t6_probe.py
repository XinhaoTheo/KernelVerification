import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_17/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
torch.manual_seed(0)
for seqlen in [128, 200, 100, 65, 63]:
    x = torch.randn(4, seqlen, device="cuda", dtype=torch.float32)
    out = m.chunked_cumsum(x, chunk=64)
    ref = torch.cumsum(x, dim=1)
    err = (out - ref).abs()
    tail = seqlen % 64
    entry = {
        "seqlen": seqlen,
        "tail_len": tail,
        "max_abs_err": float(err.max()),
        "max_abs_err_head": float(err[:, :seqlen - tail].max()) if seqlen - tail > 0 else None,
        "max_abs_err_tail": float(err[:, seqlen - tail:].max()) if tail > 0 else None,
        "tail_out_all_zero": bool(torch.all(out[:, seqlen - tail:] == 0).item()) if tail > 0 else None,
        "ref_tail_absmax": float(ref[:, seqlen - tail:].abs().max()) if tail > 0 else None,
        "allclose": bool(torch.allclose(out, ref, atol=1e-4, rtol=1e-4)),
    }
    res[str(seqlen)] = entry
print(json.dumps({"metric":"elementwise abs err vs torch.cumsum, split head/tail", "results":res}, indent=1))
