import torch, json, sys
sys.path.insert(0, "/root/cases/case_150")
import kernel

results = []
for batch in (1, 7, 128):
    for seed in (0, 1, 2):
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randint(0, 3329, (batch, 128), generator=g, dtype=torch.int32).cuda()
        out = kernel.run(x)
        ref = kernel.reference(x)
        mismatch = (out != ref).sum().item()
        results.append({"batch": batch, "seed": seed, "mismatch": mismatch, "total": out.numel()})
print(json.dumps({"N": 128, "trials": results,
                  "any_mismatch": any(r["mismatch"] > 0 for r in results)}))