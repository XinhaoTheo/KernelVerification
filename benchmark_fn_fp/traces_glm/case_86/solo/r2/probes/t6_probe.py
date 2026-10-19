import torch, json, sys
sys.path.insert(0, "/root/cases/case_159")
import kernel

dev = "cuda"
mismatches = []
tests = 0
maxdiff = 0

def make(B, N, seed, extreme=False):
    g = torch.Generator("cpu").manual_seed(seed)
    m = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
    if extreme:
        b = torch.full((B, N, 2), -32768, dtype=torch.int16)
        i = torch.full((B, 2), 32767, dtype=torch.int16)
    else:
        b = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16)
        i = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16)
    return m.to(dev), b.to(dev), i.to(dev)

for N in range(1, 33):
    for seed in range(4):
        B = [1, 2, 3, 7, 64][seed % 5]
        m, b, i = make(B, N, seed * 977 + N)
        out = kernel.run(m, b, i)
        ref = kernel.reference(m, b, i)
        tests += 1
        if not torch.equal(out, ref):
            d = (out - ref).abs().max().item()
            maxdiff = max(maxdiff, d)
            mismatches.append({"B": B, "N": N, "seed": seed, "maxdiff": d})

# extreme values & edge shapes
for (B, N) in [(1,1),(1,32),(64,1),(64,32),(5,9),(5,17),(3,25)]:
    for ext in [False, True]:
        m, b, i = make(B, N, 1234 + B * 31 + N, extreme=ext)
        out = kernel.run(m, b, i)
        ref = kernel.reference(m, b, i)
        tests += 1
        if not torch.equal(out, ref):
            mismatches.append({"B": B, "N": N, "extreme": ext, "maxdiff": (out - ref).abs().max().item()})

# deterministic adversarial: all-ones matrix amplifies, worst-case growth N=32
m = torch.ones(1, 32, 2, 2, dtype=torch.int8).to(dev)
b = torch.full((1, 32, 2), 32767, dtype=torch.int16).to(dev)
i = torch.full((1, 2), -32768, dtype=torch.int16).to(dev)
out = kernel.run(m, b, i); ref = kernel.reference(m, b, i)
tests += 1
if not torch.equal(out, ref):
    mismatches.append({"adversarial_all_ones": (out - ref).abs().max().item()})

print(json.dumps({"tests": tests, "num_mismatches": len(mismatches), "mismatches": mismatches[:5], "metric": "exact torch.equal int64 vs sequential reference"}))
