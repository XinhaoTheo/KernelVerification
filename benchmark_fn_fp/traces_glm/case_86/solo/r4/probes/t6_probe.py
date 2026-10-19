
import sys, json
sys.path.insert(0, "/root/cases/case_159")
import torch
from kernel import run, reference, make_inputs

dev = "cuda"
mismatches = []
for seed in range(3):
    for N in range(1, 33):
        B = [1, 7, 64][seed % 3]
        m, b, i = make_inputs(device=dev, batch=B, length=N, seed=seed*100+N)
        out = run(m, b, i)
        ref = reference(m, b, i)
        if not torch.equal(out, ref):
            bad = (out != ref)
            mismatches.append((seed, B, N, bad.sum().item()))
# extreme values test: all +/-1 matrices, extremes of int16 bias/initial
for N in [1, 7, 8, 9, 16, 31, 32]:
    for fill in [-1, 1]:
        B = 3
        m = torch.full((B, N, 2, 2), fill, dtype=torch.int8, device=dev)
        b = torch.tensor([[[-32768, 32767]]*N]*B, dtype=torch.int16, device=dev)
        i = torch.tensor([[-32768, 32767]]*B, dtype=torch.int16, device=dev)
        out = run(m, b, i)
        ref = reference(m, b, i)
        if not torch.equal(out, ref):
            mismatches.append(("extreme", B, N, (out != ref).sum().item()))
# growth test: matrices all 1 to maximize magnitude
for N in [16, 32]:
    m = torch.ones((1, N, 2, 2), dtype=torch.int8, device=dev)
    b = torch.full((1, N, 2), 32767, dtype=torch.int16, device=dev)
    i = torch.full((1, 2), 32767, dtype=torch.int16, device=dev)
    out = run(m, b, i); ref = reference(m, b, i)
    if not torch.equal(out, ref):
        mismatches.append(("growth", 1, N, (out != ref).sum().item()))
print(json.dumps({"mismatches": mismatches, "count": len(mismatches)}))
