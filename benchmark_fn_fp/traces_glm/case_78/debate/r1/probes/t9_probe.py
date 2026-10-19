import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel
dev = "cuda"
results = {}
for N in (64, 128):
    # delta rows: one-hot per position — exposes any permutation scrambling
    B = 4
    rows = []
    for r in range(B):
        base = [0]*N
        base[(r*17) % N] = 1
        rows.append(base)
    # random rows
    g = torch.Generator().manual_seed(1234+N)
    rand = torch.randint(0, 3329, (B, N), generator=g, dtype=torch.int32)
    x = torch.tensor(rows, dtype=torch.int32, device=dev)
    for xx, name in ((x, "delta"), (rand.to(dev), "random")):
        out = kernel.run(xx)
        ref = kernel.reference(xx.cpu()).to(dev)
        mism = int((out != ref).sum())
        results[f"N{N}_{name}"] = mism
        if mism and "first_example" not in results:
            idx = (out != ref).nonzero()[0].tolist()
            results["first_example"] = {"row": idx[0], "k": idx[1],
                "out": int(out[idx[0], idx[1]]), "ref": int(ref[idx[0], idx[1]])}
print(json.dumps({"metric": "exact_equality_mismatch_count", "results": results}))