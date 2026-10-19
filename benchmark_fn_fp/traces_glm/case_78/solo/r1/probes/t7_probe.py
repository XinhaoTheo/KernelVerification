import torch, json, importlib.util, sys, os
sys.path.insert(0, "/root/cases/case_151")
import kernel

def check(batch, width, x):
    out = kernel.run(x)
    ref = kernel.reference(x.cpu().cuda())
    ok = torch.equal(out, ref)
    return ok, int((out != ref).sum().item())

results = {}
for width in (64, 128):
    ok_all = True
    mism = 0
    for seed in range(5):
        x = torch.randint(0, 3329, (8, width), dtype=torch.int32, device="cuda")
        ok, m = check(8, width, x)
        ok_all &= ok; mism += m
    # adversarial: all-3328, all-0, all-1, random extreme
    for fill in (3328, 0, 1, 3327):
        x = torch.full((4, width), fill, dtype=torch.int32, device="cuda")
        ok, m = check(4, width, x)
        ok_all &= ok; mism += m
    # large batch boundary
    x = torch.randint(0, 3329, (128, width), dtype=torch.int32, device="cuda")
    ok, m = check(128, width, x)
    ok_all &= ok; mism += m
    # single row
    x = torch.randint(0, 3329, (1, width), dtype=torch.int32, device="cuda")
    ok, m = check(1, width, x)
    ok_all &= ok; mism += m
    results[f"N{width}"] = {"all_exact": ok_all, "total_mismatches": mism}

print(json.dumps({"metric": "exact equality (torch.equal) vs reference", "results": results}))