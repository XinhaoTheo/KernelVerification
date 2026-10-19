import sys, json
sys.path.insert(0, "/root/cases/case_151")
import torch
from kernel import run, reference, make_inputs

results = {}
mismatch_total = 0
for seed in range(6):
    x, = make_inputs(device="cuda", batch=16, width=128, seed=seed)
    out = run(x)
    ref = reference(x)
    mism = (out != ref)
    mismatch_total += int(mism.sum())
    if seed == 0 and mism.any():
        idx = mism.nonzero()[0].tolist()
        b, k = idx
        results["example"] = {"row": b, "k": k, "kernel": out[b, k].item(), "ref": ref[b, k].item()}

# also edge batches: B=1 and B=128, plus all-ones row and boundary values
x_edge = torch.ones(1, 128, dtype=torch.int32, device="cuda") * 3328
out = run(x_edge); ref = reference(x_edge)
edge_mism = int((out != ref).sum())
xb = torch.randint(0, 3329, (128, 128), dtype=torch.int32, device="cuda")
out = run(xb); ref = reference(xb)
b128_mism = int((out != ref).sum())

print(json.dumps({"metric": "exact int32 mismatch count vs reference DFT, N=128",
 "reason": "tol-0 contract; N=128 exercises odd-LOG_N radix-2 stage",
 "random_mismatch_total": mismatch_total, "edge_all_3328_mismatch": edge_mism,
 "batch128_mismatch": b128_mism, "n_tested": 128, **results}))