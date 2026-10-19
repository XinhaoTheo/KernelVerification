import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_151/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

results = {}
mismatch_total = 0
for width in (64, 128):
    for batch in (1, 7, 128):
        for seed in (0, 1, 2, 123):
            g = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.randint(0, 3329, (batch, width), generator=g, dtype=torch.int32).cuda()
            out = k.run(x)
            ref = k.reference(x)
            mism = int((out != ref).sum().item())
            mismatch_total += mism
            results[f"N{width}_B{batch}_s{seed}"] = {"equal": mism == 0, "mismatches": mism}
for width in (64, 128):
    for fill in (0, 3328):
        x = torch.full((3, width), fill, dtype=torch.int32, device="cuda")
        mism = int((k.run(x) != k.reference(x)).sum().item())
        mismatch_total += mism
        results[f"N{width}_const{fill}"] = {"equal": mism == 0, "mismatches": mism}
print(json.dumps({"exact_all": mismatch_total == 0, "mismatch_elements": mismatch_total, "details": results}))
