import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_151/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

results = {}
mismatch_total = 0
for width in (64, 128):
    for batch in (1, 7, 128):
        for seed in (0, 1, 2, 123):
            x = torch.randint(0, 3329, (batch, width), dtype=torch.int32, device="cuda")
            out = k.run(x)
            ref = k.reference(x)
            eq = torch.equal(out, ref)
            mismatch_total += int((~eq).sum().item()) if not eq else 0
            results[f"N{width}_B{batch}_s{seed}"] = bool(eq)
# edge cases: all-zero and all-max rows
for width in (64, 128):
    for fill in (0, 3328):
        x = torch.full((3, width), fill, dtype=torch.int32, device="cuda")
        eq = torch.equal(k.run(x), k.reference(x))
        results[f"N{width}_const{fill}"] = bool(eq)
        if not eq: mismatch_total += 1
print(json.dumps({"exact_all": mismatch_total == 0, "mismatch_elements": mismatch_total, "details": results}))
