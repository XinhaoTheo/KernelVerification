import torch, json, sys
sys.path.insert(0, "/root/cases/case_151")
import kernel

res = {}
for width in (64, 128):
    mism = 0
    worst = 0
    for trial in range(10):
        x = torch.randint(0, 3329, (4, width), dtype=torch.int32, device="cuda")
        out = kernel.run(x)
        ref = kernel.reference(x)
        d = (out != ref).sum().item()
        mism += int(d)
        worst = max(worst, int(d))
    res[f"N{width}"] = {"batches": 10, "batch_rows": 4, "total_mismatches": mism, "worst_single_batch_mismatches": worst}
print(json.dumps({"metric": "exact equality (torch.equal) vs reference on random canonical inputs", "results": res}))