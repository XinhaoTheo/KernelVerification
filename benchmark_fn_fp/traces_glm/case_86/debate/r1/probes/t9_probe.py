import json, torch, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel

res = {"cases": 0, "mismatch": 0, "errors": []}
# N<=8: padded=1, tree is trivial -> only _local_scan + apply matter.
for N in range(1, 9):
    for seed in range(5):
        m, b, i = kernel.make_inputs(device="cuda", batch=4, length=N, seed=seed)
        out = kernel.run(m, b, i)
        ref = kernel.reference(m, b, i)
        res["cases"] += 1
        if not torch.equal(out, ref):
            res["mismatch"] += 1
            if len(res["errors"]) < 3:
                res["errors"].append({"N": N, "seed": seed})
        # identity-padded tail lanes: N=1..7 exercises padded lanes with identity payloads
print(json.dumps({**res, "supports_defect": res["mismatch"] > 0}))