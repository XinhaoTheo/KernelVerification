import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel
dev = "cuda"
results = {}
for N in (64, 128):
    ones = torch.ones((2, N), dtype=torch.int32, device=dev) * 1
    # delta at position 1 exercises omega powers directly
    d = torch.zeros((1, N), dtype=torch.int32, device=dev); d[0, 1] = 1
    # varied values row
    g = torch.Generator().manual_seed(99)
    var = torch.randint(0, 3329, (3, N), generator=g, dtype=torch.int32).to(dev)
    for xx, name in ((ones, "ones"), (d, "delta1"), (var, "varied")):
        out = kernel.run(xx)
        ref = kernel.reference(xx.cpu()).to(dev)
        results[f"N{N}_{name}"] = int((out != ref).sum())
print(json.dumps({"metric": "exact_equality_mismatch_count", "results": results}))