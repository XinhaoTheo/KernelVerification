import sys, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_159/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
mismatches = []
shapes_tested = 0
for N in range(1, 33):
    for B, seed in [(1, 0), (5, 1), (17, 2), (64, 3)]:
        g = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
        bias = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16)
        initial = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16)
        matrix, bias, initial = matrix.to(dev), bias.to(dev), initial.to(dev)
        out = k.run(matrix, bias, initial)
        ref = k.reference(matrix, bias, initial)
        shapes_tested += 1
        if not torch.equal(out, ref):
            d = (out != ref)
            mismatches.append({"B": B, "N": N, "seed": seed, "num_mismatched_elements": int(d.sum())})
# extreme-value stress: all-ones matrices, extreme biases/initials, every N
for N in range(1, 33):
    B = 4
    matrix = torch.ones(B, N, 2, 2, dtype=torch.int8, device=dev)
    bias = torch.where(torch.arange(N * 2, device=dev).view(1, N, 2) % 2 == 0,
                       torch.full((B, N, 2), -32768, dtype=torch.int16, device=dev),
                       torch.full((B, N, 2), 32767, dtype=torch.int16, device=dev))
    initial = torch.tensor([[32767, -32768]] * B, dtype=torch.int16, device=dev)
    out = k.run(matrix, bias, initial); ref = k.reference(matrix, bias, initial)
    shapes_tested += 1
    if not torch.equal(out, ref):
        mismatches.append({"B": B, "N": N, "case": "extreme_ones", "num": int((out != ref).sum())})
print(json.dumps({"mismatch_cases": mismatches, "shapes_tested": shapes_tested, "all_exact": len(mismatches) == 0}))