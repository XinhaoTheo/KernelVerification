import torch, json, traceback, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_13/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res = {}
try:
    kv = torch.randn(2, 96, device="cuda")
    out = m.gqa_gather(kv, 4)
    torch.cuda.synchronize()
    # check pow2 behavior: does it run? correct rows per contract?
    n_rep = 4 // 2
    ref = torch.stack([kv[q // n_rep] for q in range(4)])
    res["compiled_ran"] = True
    res["exact_match_consecutive_contract"] = bool(torch.equal(out, ref))
    res["mismatched_rows"] = [q for q in range(4) if not torch.equal(out[q], ref[q])]
except Exception as e:
    res["compiled_ran"] = False
    res["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(res))