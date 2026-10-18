import torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_158/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(0)
results = {}
# make_inputs default
m, b, i = k.make_inputs(device=dev, batch=7, length=32, seed=0)
out = k.run(m, b, i); ref = k.reference(m, b, i)
results["make_inputs_default_B7_N32"] = bool(torch.equal(out, ref))
# random sweep across shapes
mismatches = []
for t in range(30):
    B = int(torch.randint(1, 65, (1,), generator=g))
    N = int(torch.randint(1, 33, (1,), generator=g))
    m = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8).to(dev)
    b = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16).to(dev)
    i = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16).to(dev)
    out = k.run(m, b, i); ref = k.reference(m, b, i)
    if not torch.equal(out, ref):
        mismatches.append({"B": B, "N": N, "max_abs_diff": (out-ref).abs().max().item()})
results["random_30_cases_mismatch_count"] = len(mismatches)
print({"results": results, "mismatches": mismatches[:5], "claim": "c3", "metric": "bitwise exact int64 equality vs reference across make_inputs default and 30 random (B,N) cases — tests tuple associative_scan and tl.gather semantics in Triton 3.4"})