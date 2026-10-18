import json, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel, torch
res = {}
for N in [9, 12, 16, 17, 20, 24]:
    m, b, i = kernel.make_inputs(device="cuda", batch=4, length=N, seed=100 + N)
    try:
        out = kernel.run(m, b, i)
        ref = kernel.reference(m, b, i)
        diff = (out != ref)
        chunk_of_first = int(torch.nonzero(diff, as_tuple=False).view(-1)[0].item() // N) if diff.any() else -1
        res[f"N{N}"] = {"exact": bool(not diff.any().item()), "chunk_of_first_diff": chunk_of_first, "n_mismatch": int(diff.sum().item())}
    except Exception as e:
        res[f"N{N}"] = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps({"claim": "c2", "probe": "exact vs reference for chunks=2 (N=9..16) and chunks=3 (N=17..24)", "results": res}))