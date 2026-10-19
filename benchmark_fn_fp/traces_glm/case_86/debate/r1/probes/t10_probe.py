import json, torch, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel, triton

res = {"cases": 0, "mismatch_by_chunks": {}, "errors": []}
for N in [8, 9, 15, 16, 17, 24, 25, 32]:  # chunks 1..4, CHUNKS<padded at 9..15(2),17..24(3)
    for seed in range(8):
        m, b, i = kernel.make_inputs(device="cuda", batch=4, length=N, seed=seed)
        out = kernel.run(m, b, i)
        ref = kernel.reference(m, b, i)
        chunks = triton.cdiv(N, 8)
        res["cases"] += 1
        if not torch.equal(out, ref):
            res["mismatch_by_chunks"][chunks] = res["mismatch_by_chunks"].get(chunks, 0) + 1
            if len(res["errors"]) < 3:
                bad = (out != ref)
                res["errors"].append({"N": N, "seed": seed, "chunks": chunks,
                    "first_bad_step": int(bad.any(dim=(0,2)).nonzero()[0].item()),
                    "n_bad_elems": int(bad.sum().item())})
print(json.dumps({**res, "supports_defect": len(res["mismatch_by_chunks"]) > 0}))